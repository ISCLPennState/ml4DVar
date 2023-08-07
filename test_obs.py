import os, sys
sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from torch.utils.data import IterableDataset, DataLoader
import torch
#from d2l import torch as d2l
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
from src.dv import *
import time
from scipy.interpolate import interpn

#from climax.global_forecast_4dvar.train import main

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

NUM_MODEL_STEPS=2 #Current climax works at 6 hours so with assimilation wind of 12 hours we need to call the model twice

class ObsError(torch.nn.Module):
    def __init__(self, vars, var_types, var_obs_errs, obs_perc_errs, var_stds):
        super().__init__()
        mult_vars = np.zeros(len(vars), dtype = bool)
        obs_err = np.zeros(len(vars), dtype = 'f4')
        for var_type, var_obs_err, obs_perc_err in zip(var_types, var_obs_errs, obs_perc_errs):
            var_idxs = [i for i, var in enumerate(vars) if var_type in var]
            if not obs_perc_err:
                obs_err[var_idxs] = (var_obs_err / np.array([var_stds[var][0] for var in vars if var_type in var]))**2.0
            else:
                obs_err[var_idxs] = var_obs_err
            mult_vars[var_idxs] = obs_perc_err
        self.mult_vars = torch.from_numpy(mult_vars).to(device)
        self.obs_err = torch.from_numpy(obs_err).to(device)
    def forward(self, obs, x_obs, var):
        if self.mult_vars[var]:
            return torch.sum((x_obs - obs)**2.0 / (self.obs_err[var] * torch.abs(obs)) ** 2.0)
        else:
            #if var > 2 and var < 7:
                #print(var)
                #print((x_obs - obs)**2.0)
            return torch.sum((x_obs - obs)**2.0) / self.obs_err[var]

def observe_linear(x, H_idxs, H_vals):
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1),
                       axis = 1).to(device)
    return output

def threeDresidual(x, *args):
    '''
    :param x: Initial condition for the forward model. [num_vars, 128, 256]
    :param args:
    (
        background - [num_vars, 128, 256]
        background_error - [num_vars, 128] array containing estimated background error covariance of SH components
        obs - [time_steps, num_vars, Max_num_obs]. Each [Max_num_obs] array is padded with zeros up to the
            maximum number of observations
        H_idxs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        H_obs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        num_obs - [time_steps, num_vars]
        obs_err - ObsError layer
        nn_model - ClimaX model
        dv_layer - NN Layer for converting u, v wind to divergence, vorticity
        sht - RealSHT object for computing spherical harmonics
        sht_scaler - Matrix for scaling sht coefficients to account for 1-sided sht

        For the 3D-Var cycle, time_steps must be 1

    )
    :return:
    '''
    background = args[0]
    background_err = args[1]
    background_err_hf = args[2]
    obs = args[3].to(device)
    H_idxs = args[4]
    H_obs = args[5]
    num_obs = args[6]
    obs_err = args[7]
    nn_model = args[8]
    dv_layer = args[9]
    sht = args[10]
    inv_sht = args[11]
    sht_scaler = args[12]
    print_loss = args[13]
    save_loss_comps = args[14]

    vars = ['2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'geopotential_500',
            'geopotential_700',
            'geopotential_850',
            'geopotential_925',
            'u_component_of_wind_250',
            'u_component_of_wind_500',
            'u_component_of_wind_700',
            'u_component_of_wind_850',
            'u_component_of_wind_925',
            'v_component_of_wind_250',
            'v_component_of_wind_500',
            'v_component_of_wind_700',
            'v_component_of_wind_850',
            'v_component_of_wind_925',
            'temperature_250',
            'temperature_500',
            'temperature_700',
            'temperature_850',
            'temperature_925',
            'specific_humidity_250',
            'specific_humidity_500',
            'specific_humidity_700',
            'specific_humidity_850',
            'specific_humidity_925']


    num_vars = x.shape[0]
    time_steps = obs.shape[0]

    #total_obs = torch.sum(num_obs)

    # Compute background error with identity background error covariance
    dvx = dv_layer(x).to(device)
    dvb = dv_layer(background).to(device)
    diff = dvx - dvb
    coeff_diff = sht(diff)

    se_background_comps_unscaled = torch.abs(coeff_diff.to(device) * torch.conj(coeff_diff.to(device)))
    #print(se_background_comps_unscaled.shape)
    #for i in range(se_background_comps_unscaled.shape[0]):
    #    print(se_background_comps_unscaled[i])
    #print(torch.sum(se_background_comps_unscaled * sht_scaler))
    #print(torch.sum((dvx - dvb)**2.0))
    #print(torch.sum((x - background)**2.0))
    se_background_comps = torch.sum(se_background_comps_unscaled / torch.unsqueeze(background_err.to(device), 2) * \
        sht_scaler, (1,2))
    se_background = torch.sum(se_background_comps)
    if print_loss:
        print('Background')
        print(se_background)

    hf_diff = diff - inv_sht(coeff_diff)
    se_background_hf_comps = torch.sum(torch.abs(hf_diff)**2.0 / background_err_hf, (1,2))
    se_background_hf = torch.sum(se_background_hf_comps)
    if print_loss:
        print('Background HF')
        print(se_background_hf)

    if save_loss_comps:
        save_array = np.zeros(3*len(vars))
        save_array[:len(vars)] = se_background_comps.detach().cpu().numpy()
        save_array[len(vars):2*len(vars)] = se_background_hf_comps.detach().cpu().numpy()

    #Compute error in observations at first time step for all variables
    se_obs = torch.zeros(1).to(device)
    #print(x.shape)
    for var in range(num_vars):
        # Form sparse observation matrix from inputs. This matrix has a shape of (num_obs, nlat * nlon), and each row
        # has 4 elements corresponding to the 4 surrounding grid points on the lat - lon grid that are being
        # interpolated
        # H = torch.sparse_coo_tensor((torch.arange(num_obs[0, var]).reshape(-1, 1).expand(num_obs[0, var], 4),
        #                              H_idxs[0, var, :num_obs[0, var]*4]), H_obs[0, var, :num_obs[0, var]*4],
        #                             (num_obs[0, var], x.shape[1]*x.shape[2])).to_sparse_csr()
        x_obs = observe_linear(x[var].reshape(-1, 1),
                               H_idxs[0, var, :4*num_obs[0, var]].reshape(-1, 4).T,
                               H_obs[0, var, :4*num_obs[0, var]].reshape(-1, 4))
        var_err = obs_err(x_obs, obs[0, var, :num_obs[0, var]], var)
        if print_loss:
            print(vars[var])
            print(var_err)
            #print('H_idxs')
            #print(H_idxs[0, var, :4*num_obs[0, var]])
            #print('H_weights')
            #print(H_obs[0, var, :4*num_obs[0, var]])
            #print('Obs')
            #print(obs[0, var, :num_obs[0, var]])
            #print('X_Obs')
            #print(x_obs)
        if save_loss_comps:
            save_array[var+2*len(vars)] = var_err.detach().cpu().numpy()
        se_obs += var_err
    #print(se_obs)
    #print(se_background)
    if save_loss_comps:
        return se_background + se_background_hf + se_obs, save_array
    else:
        return se_background + se_background_hf + se_obs

def fourDresidual(x, *args):
    '''
    :param x: Initial condition for the forward model. [num_vars, 128, 256]
    :param args:
    (
        background - [num_vars, 128, 256]
        background_error - [num_vars, 128] array containing estimated background error covariance of SH components
        obs - [time_steps, num_vars, Max_num_obs]. Each [Max_num_obs] array is padded with zeros up to the
            maximum number of observations
        H_idxs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        H_obs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        num_obs - [time_steps, num_vars]
        obs_err - ObsError layer
        nn_model - ClimaX model
        dv_layer - NN Layer for converting u, v wind to divergence, vorticity
        sht - RealSHT object for computing spherical harmonics
        sht_scaler - Matrix for scaling sht coefficients to account for 1-sided sht

    )
    :return:
    '''
    background = args[0]
    background_err = args[1]
    obs = args[2].to(device)
    H_idxs = args[3]
    H_obs = args[4]
    num_obs = args[5]
    obs_err = args[6]
    nn_model = args[7]
    dv_layer = args[8]
    sht = args[9]
    sht_scaler = args[10]
    
    num_vars = x.shape[0]
    time_steps = obs.shape[0]
    #total_obs = torch.sum(num_obs)

    # Compute background error with identity background error covariance
    dvx = dv_layer(x).to(device)
    dvb = dv_layer(background).to(device)
    coeff_diff = sht(dvx - dvb)
    se_background = torch.sum((torch.abs(coeff_diff.to(device) * torch.conj(coeff_diff.to(device))) / torch.unsqueeze(background_err.to(device), 2)) * \
        sht_scaler)

    #Compute error in observations at first time step for all variables
    se_obs = torch.zeros(1).to(device)
    for var in range(num_vars):
        # Form sparse observation matrix from inputs. This matrix has a shape of (num_obs, nlat * nlon), and each row
        # has 4 elements corresponding to the 4 surrounding grid points on the lat - lon grid that are being
        # interpolated
        # H = torch.sparse_coo_tensor((torch.arange(num_obs[0, var]).reshape(-1, 1).expand(num_obs[0, var], 4),
        #                              H_idxs[0, var, :num_obs[0, var]*4]), H_obs[0, var, :num_obs[0, var]*4],
        #                             (num_obs[0, var], x.shape[1]*x.shape[2])).to_sparse_csr()
        x_obs = observe_linear(x.reshape(-1, 1),
                               H_idxs[0, var, :4*num_obs[0, var]].reshape(-1, 4).T,
                               H_obs[0, var, :4*num_obs[0, var]].reshape(-1, 4))
        se_obs += obs_err(x_obs, obs[0, var, :num_obs[0, var]], var)
        #se_obs += torch.sum((x_obs - obs[0, var, :num_obs[0, var]]) ** 2.0) / obs_err

    # Compute effects on later observations
    #assert(time_steps == 2)
    se_obs_2 = torch.zeros(1).to(device)
    for step in range(1, time_steps):
        # Update model state using forward model
        #temp = torch.clone(x)
        #with torch.inference_mode():
        print(x.size)
        temp = torch.clone(x.to(device))
        with torch.inference_mode():
            nn_model.full_input = torch.cat((nn_model.full_input[:,:4,:,:],torch.unsqueeze(temp,dim=0)),dim=1)
            _, x = nn_model.net.forward_multi_step(nn_model.full_input, temp, nn_model.lead_times, nn_model.hold_variables, nn_model.hold_out_variables,steps=NUM_MODEL_STEPS) #x = nn_model.forward(x) #_, x = nn_model.net.forward_multi_step(nn_model.full_input, x, nn_model.lead_times, nn_model.hold_variables, nn_model.hold_out_variables,steps=NUM_MODEL_STEPS)
        print('x.requires_grad',x.requires_grad)
        plotting_vars = x.detach().cpu().numpy()
        #np.save('/eagle/MDClimSim/awikner/climax_4dvar_troy/test_pred.npy', plotting_vars)
        #print(plotting_vars[0,0,:,:])
        #plt.pcolormesh(plotting_vars[0,0,:,:])
        #plt.colorbar()
        #plt.show()
        for var in range(num_vars):
            x_obs = observe_linear(x.reshape(-1, 1),
                                   H_idxs[step, var, :4 * num_obs[step, var]].reshape(-1, 4).T.to(device),
                                   H_obs[step, var, :4 * num_obs[step, var]].reshape(-1, 4).to(device))

            #print(x_obs)
            #print(obs[step, var, :num_obs[step, var]])
            var_err = obs_err(x_obs.to(device), obs[step, var, :num_obs[step, var]].to(device), var)
            #print(var_err)
            se_obs_2 += var_err
            #se_obs += torch.sum((x_obs - obs[step, var, :num_obs[step, var]]) ** 2.0) / obs_err
    print(se_obs + se_obs_2)
    print(se_background)
    #sys.exit(2)

    return se_obs + se_background

class ObsDataset(IterableDataset):
    def __init__(self, file_path, start_datetime, end_datetime, window_len, window_step, model_step, vars):
        super().__init__()
        self.save_hyperparameters()
        datetime_diff = end_datetime - start_datetime
        hour_diff = datetime_diff.days*24 + datetime_diff.seconds // 3600
        self.all_obs_datetimes = [start_datetime + timedelta(hours = i) for i in \
                             range(0, hour_diff + model_step, model_step)]
        self.window_len_idxs = window_len // model_step + 1
        self.window_step_idxs = window_step // model_step
        self.num_cycles = (len(self.all_obs_datetimes) - self.window_len_idxs) // self.window_step_idxs

    def read_file(self):
        with h5py.File(self.file_path, 'r') as f:
            obs_datetimes = self.all_obs_datetimes[self.window_start: self.window_start + self.window_len_idxs]
            print('Obs. Datetimes')
            print(obs_datetimes)
            shapes = np.zeros((self.window_len_idxs, len(self.vars)), dtype = int)
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                shapes[i, j] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var].shape[0]
            max_obs = np.max(shapes)
            all_obs = np.zeros((self.window_len_idxs, len(self.vars), max_obs))
            H_idxs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs), dtype = 'i4')
            H_obs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs))
            obs_latlon = np.zeros((self.window_len_idxs, len(self.vars), max_obs, 2))
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                all_obs[i, j, :shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 2]
                H_idxs[i, j, :4*shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 0]
                H_obs[i, j, :4 * shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 1]
                obs_latlon[i, j, :shapes[i,j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, :2]
            output = (torch.from_numpy(all_obs).to(device), torch.from_numpy(H_idxs).long().to(device), torch.from_numpy(H_obs).to(device),
                      torch.from_numpy(shapes).long().to(device), obs_latlon)
            return output

    def __iter__(self):
        self.window_start = -self.window_step_idxs
        return self

    def __next__(self):
        if self.window_start <= len(self.all_obs_datetimes) - self.window_len_idxs:
            self.window_start += self.window_step_idxs
            return self.read_file()
        else:
            raise StopIteration

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class FourDVar():
    def __init__(self, nn_model, obs_dataloader, background, background_err, background_err_hf, obs_err, dv_layer, lr = 1.,
                 max_iter = 700):
        super(FourDVar).__init__()
        self.save_hyperparameters()
        #self.board = d2l.ProgressBoard(xlabel = '4D Var loss', ylabel = 'Cycle Iteration',
        #                               xlim = [0, obs_dataloader.num_cycles])
        self.sht = th.RealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.inv_sht = th.InverseRealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2)).reshape(1, 1, -1).to(device)
        self.save_dir = '/eagle/MDClimSim/awikner/climax_4dvar_troy/data/climaX'
        
    def loss(self, print_loss = False, save_loss_comps = False, itr = 0):
        #print(self.H_idxs.size())
        if self.H_idxs.size(dim = 1) == 1:
            out = threeDresidual(self.x[0].to(device),
                             self.background[0].to(device),
                             self.background_err.to(device),
                             self.background_err_hf.to(device),
                             self.all_obs[0],
                             self.H_idxs[0],
                             self.H_obs[0],
                             self.n_obs[0],
                             self.obs_err,
                             self.nn_model,
                             self.dv_layer,
                             self.sht,
                             self.inv_sht,
                             self.sht_scaler,
                             print_loss,
                             save_loss_comps)
            if save_loss_comps:
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d.npy' % (itr, self.step))
                np.save(filename, out[1])
                return out[0]
            else:
                return out
        else:
            return fourDresidual(self.x[0].to(device),
                                 self.background[0].to(device),
                                 self.background_err.to(device),
                                 self.all_obs[0],
                                 self.H_idxs[0],
                                 self.H_obs[0],
                                 self.n_obs[0],
                             self.obs_err,
                             self.nn_model,
                             self.dv_layer,
                             self.sht,
                             self.sht_scaler)

    def configure_optimizer(self):
        return torch.optim.LBFGS([self.x], lr = self.lr, max_iter = self.max_iter, history_size=300, tolerance_grad = 1e-5)
        #return torch.optim.SGD([self.x], lr = self.lr)

    def cycle(self, itr):
        self.step = 0
        def closure():
            self.optim.zero_grad()
            self.nn_model.zero_grad()
            loss = self.loss(print_loss = False, save_loss_comps = True, itr = itr)
            #self.optim.zero_grad()
            #self.nn_model.zero_grad()
            self.step += 1
            #print(loss)
            #sys.exit(2)
            loss.backward(retain_graph=False)
            return loss
         
        #init_loss, loss_comps = self.loss(print_loss = True, save_loss_comps = True)
        #np.save('/eagle/MDClimSim/awikner/climax_4dvar_troy/data/init_loss_comps_%d.npy' % itr,
        #        loss_comps)
        #for itr in range(self.max_iter):
        #    self.optim.zero_grad()
        #    self.nn_model.zero_grad()
        #    if itr < 2:
        #        loss = self.loss(print_loss = True)
        #    else:
        #        loss = self.loss()
        #    print('Iter %d loss: %e' % (itr, loss))
        #    loss.backward(retain_graph=False)
        #    self.optim.step()
        print(torch.sum(self.background))
        np.save(os.path.join(self.save_dir, 'background_%d.npy' % itr), background.detach().cpu().numpy())
        self.optim.step(closure)
        save_analysis = self.x.detach().cpu().numpy()
        np.save(os.path.join(self.save_dir, 'analysis_%d.npy' % itr), save_analysis)
        cycle_loss = self.loss(print_loss = True, save_loss_comps = False)
        #np.save('/eagle/MDClimSim/awikner/climax_4dvar_troy/data/final_loss_comps_%d.npy' % itr,
        #        loss_comps)
        #print('Cycle loss: %e' % cycle_loss)
        temp = torch.clone(self.x)
        for i in range(self.obs_dataloader.dataset.window_step_idxs):
            with torch.inference_mode():
                nn_model.full_input[0,4::,:,:] = temp
                _, temp = nn_model.net.forward_multi_step(nn_model.full_input, temp, nn_model.lead_times, nn_model.hold_variables, nn_model.hold_out_variables,steps=NUM_MODEL_STEPS)
        self.background = torch.clone(temp.detach())
        self.x = torch.clone(temp.detach())
        self.x.requires_grad_(True)
        return self.background, cycle_loss
        #self.plot('loss', cycle_loss, False)

    def fourDvar(self):
        self.x = torch.clone(self.background)
        #self.x = torch.from_numpy(np.load(os.path.join(self.save_dir, 'analysis_0.npy'))).to(device)
        self.x.requires_grad_(True)
        for itr, (self.all_obs, self.H_idxs, self.H_obs, self.n_obs) in enumerate(self.obs_dataloader):
            self.optim = self.configure_optimizer()
            _, cycle_loss = self.cycle(itr)
            print('Cycle loss %d: %0.2f' % (itr, cycle_loss))

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


if __name__ == '__main__':
    filepath = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"
    means = np.load('/eagle/MDClimSim/awikner/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/awikner/normalize_std.npz')
    dv_param_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'
    background_err_file = '/eagle/MDClimSim/awikner/background_err_sh_coeffs_var.npy'
    background_err_hf_file = '/eagle/MDClimSim/awikner/background_err_hf_var.npy'
    background_file = '/eagle/MDClimSim/awikner/2014_0.npz'
    #filepath = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\irga_1415_test1_obs.hdf5'
    #means = np.load('C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\normalize_mean.npz')
    #stds = np.load('C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\normalize_std.npz')
    #dv_param_file = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\dv_params_128_256.hdf5'
    #background_err_file = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\background_err_sh_coeffs_std.npy'


    vars = ['2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'geopotential_500',
            'geopotential_700',
            'geopotential_850',
            'geopotential_925',
            'u_component_of_wind_250',
            'u_component_of_wind_500',
            'u_component_of_wind_700',
            'u_component_of_wind_850',
            'u_component_of_wind_925',
            'v_component_of_wind_250',
            'v_component_of_wind_500',
            'v_component_of_wind_700',
            'v_component_of_wind_850',
            'v_component_of_wind_925',
            'temperature_250',
            'temperature_500',
            'temperature_700',
            'temperature_850',
            'temperature_925',
            'specific_humidity_250',
            'specific_humidity_500',
            'specific_humidity_700',
            'specific_humidity_850',
            'specific_humidity_925']



    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0]
    obs_perc_err = [False, False, False, False, False]
    obs_err = ObsError(vars, var_types, var_obs_err, obs_perc_err, stds)
    #print(obs_err.obs_err)
    dv_layer = DivergenceVorticity(vars, means, stds, dv_param_file)

    background_err = torch.from_numpy(np.load(background_err_file)).float()
    background_err = background_err[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]
    background_err_hf = torch.from_numpy(np.load(background_err_hf_file)).float()
    background_err_hf = background_err_hf[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]

    start_date = datetime(2014, 1, 1, hour = 0)
    end_date = datetime(2015, 12, 31, hour = 12)
    window_len = 0
    window_step = 12
    model_step = 12

    lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
    lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')

    obs_dataset = ObsDataset(filepath, start_date, end_date, window_len, window_step, model_step, vars)

    loader = DataLoader(obs_dataset, batch_size = 1, num_workers=0)

    background_f = np.load(background_file)
    
    era5 = np.zeros((background_f['2m_temperature'].shape[0]//12+1, len(vars), 128, 256))
    for i, var in enumerate(vars):
        era5[:, i] = (background_f[var][::12, 0] - means[var][0])/stds[var][0]

    plot_var = 3
    obs_idx = 300
    for itr, (all_obs, H_idxs, H_obs, n_obs, obs_latlon) in enumerate(loader):
        print(obs_latlon.shape)
        plot_lat = (obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 0]+90)*128/180
        plot_lon = obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 1]
        plot_lon[plot_lon < 0] = plot_lon[plot_lon < 0] + 360
        plot_lon = plot_lon * 256/360
        fig, axs = plt.subplots(1, 3, figsize = (18,5))
        pcm = axs[0].pcolormesh(era5[itr, plot_var])
        axs[0].scatter(plot_lon, plot_lat, c=all_obs[0, 0, plot_var, :n_obs[0, 0, plot_var]].detach().cpu().numpy(), s=30, edgecolor='k')
        plt.colorbar(pcm, ax = axs[0])
        print(np.max(lat))
        print(np.min(lat))
        print(np.max(lon))
        print(np.min(lon))
        print(np.max(plot_lon.numpy() * 360/256))
        print(np.min(plot_lon.numpy() * 360/256))
        print(np.max(obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 0].numpy()))
        print(np.min(obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 0].numpy()))

        lat_idx = np.sum(lat - obs_latlon[0,0,plot_var,obs_idx,0].numpy() < 0) - 1
        lon_idx = np.sum(lon - obs_latlon[0,0,plot_var,obs_idx,1].numpy() < 0) - 1
        
        print((lat_idx, lon_idx))
        print(np.unravel_index(H_idxs[0,0,plot_var,4*obs_idx].detach().cpu().numpy(), (lat.size, lon.size)))

        np.save('test_obs.npy', np.stack((obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 0].numpy(), plot_lon.numpy() * 360/256, all_obs[0, 0, plot_var, :n_obs[0, 0, plot_var]].detach().cpu().numpy()), axis = 1))
        np.save('test_era5.npy', era5[itr, plot_var])
        era5_obs = interpn((lat, lon), era5[itr, plot_var], (obs_latlon[0, 0, plot_var, :n_obs[0, 0, plot_var], 0].numpy(), plot_lon.numpy() * 360/256), bounds_error = False)
        print(era5_obs.shape)
        scp = axs[1].scatter(plot_lon, plot_lat, c=all_obs[0, 0, plot_var, :n_obs[0, 0, plot_var]].detach().cpu().numpy() - era5_obs, s=30, edgecolor='k')
        plt.colorbar(scp, ax = axs[1])
        era5_obs2 = observe_linear(torch.from_numpy(era5[itr, plot_var].reshape(-1,1)).to(device), H_idxs[0, 0, plot_var, :4*n_obs[0, 0,plot_var]].reshape(-1, 4).T , H_obs[0, 0, plot_var, :4*n_obs[0, 0,plot_var]].reshape(-1, 4))
        scp2 = axs[2].scatter(plot_lon, plot_lat, c=all_obs[0, 0, plot_var, :n_obs[0, 0, plot_var]].detach().cpu().numpy() - era5_obs2.detach().cpu().numpy(), s=30, edgecolor='k')
        plt.colorbar(scp2, ax = axs[2])
        plt.show()



    """
    tic = time.perf_counter()
    iters = 0
    num_obs = 0
    for batch in loader:
        iters += 1
        num_obs += torch.sum(batch[3])
    toc = time.perf_counter()
    avg_obs = num_obs/(3 * iters)
    print('Total time: %f sec., Total iters: %d' % ((toc - tic), iters))
    print('Avg. # of Obs. per 12 hours: %0.2f' % avg_obs)
    print('Load time per batch: %f sec.' % ((toc - tic)/iters))
    """
    # Import functions from module.py im climaX
    # batch in the form of x, y, lead_time, out_variables, in_variables
    # arch.py


