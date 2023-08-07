import os, sys
sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from torch.utils.data import IterableDataset, DataLoader
import torch
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
from src.dv import *
from src.obs import *

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

    # Compute background error with identity background error covariance
    dvx = dv_layer(x).to(device)
    dvb = dv_layer(background).to(device)
    diff = dvx - dvb
    coeff_diff = sht(diff)

    se_background_comps_unscaled = torch.abs(coeff_diff.to(device) * torch.conj(coeff_diff.to(device)))

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
    for var in range(num_vars):
        x_obs = observe_linear(x[var].reshape(-1, 1),
                               H_idxs[0, var, :4*num_obs[0, var]].reshape(-1, 4).T,
                               H_obs[0, var, :4*num_obs[0, var]].reshape(-1, 4))
        var_err = obs_err(x_obs, obs[0, var, :num_obs[0, var]], var)
        if print_loss:
            print(vars[var])
            print(var_err)
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
        x_obs = observe_linear(x.reshape(-1, 1),
                               H_idxs[0, var, :4*num_obs[0, var]].reshape(-1, 4).T,
                               H_obs[0, var, :4*num_obs[0, var]].reshape(-1, 4))
        se_obs += obs_err(x_obs, obs[0, var, :num_obs[0, var]], var)
        #se_obs += torch.sum((x_obs - obs[0, var, :num_obs[0, var]]) ** 2.0) / obs_err

    # Compute effects on later observations
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

class FourDVar():
    def __init__(self, nn_model, obs_dataloader, background, background_err, background_err_hf, obs_err, dv_layer,
                 lr = 1., max_iter = 700, forecast_steps = 20, spin_up_cycles = 9, runstr = None, save_analysis = True,
                 savedir = None, NUM_MODEL_STEPS=2):
        super(FourDVar).__init__()
        self.save_hyperparameters()
        self.sht = th.RealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.inv_sht = th.InverseRealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2)).reshape(1, 1, -1).to(device)
        if not self.save_dir:
            self.save_dir = os.path.join(os.getcwd(), 'data')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        if not self.runstr:
            self.runstr = "%dhr_%s" % (obs_dataloader.dataset.window_step, obs_dataloader.dataset.start_datetime.strftime("%m%d%Y"))
        #self.save_dir = '/eagle/MDClimSim/awikner/climax_4dvar_troy/data/climaX'
        
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
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d_%s.npy' % (itr, self.step, self.runstr))
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

    def cycle(self, itr, forecast):
        self.step = 0
        def closure():
            self.optim.zero_grad()
            self.nn_model.zero_grad()
            loss = self.loss(print_loss = False, save_loss_comps = False, itr = itr)
            self.step += 1
            print(loss)
            loss.backward(retain_graph=False)
            return loss

        if self.save_analysis:
            np.save(os.path.join(self.save_dir, 'background_%d_%s.npy' % (itr, self.runstr)),
                    self.background.detach().cpu().numpy())
        self.optim.step(closure)
        if self.save_analysis:
            save_analysis = self.x.detach().cpu().numpy()
            np.save(os.path.join(self.save_dir, 'analysis_%d_%s.npy' % (itr, self.runstr)), save_analysis)
        cycle_loss = self.loss(print_loss = True, save_loss_comps = False)

        self.background = self.run_forecast(self.x)
        self.x = torch.clone(self.background)
        self.x.requires_grad_(True)
        if forecast and itr > self.spin_up_cycles:
            self.run_forecasts(itr)
        return self.background, cycle_loss

    def run_forecasts(self, itr):
        forecasts = np.zeros((self.forecast_steps, self.background.shape[-3], self.background.shape[-2], self.background.shape[-1]))
        forecasts[0] = self.background[0].detach().cpu().numpy()
        temp = self.background
        for i in range(self.forecast_steps - 1):
            temp = self.run_forecast(temp)
            forecasts[i+1] = temp.detach().cpu().numpy()[0]
        np.save(os.path.join(self.save_dir, 'forecasts_%d_%s.npy' % (itr, self.runstr)), forecasts)
        return

    def run_forecast(self, ic):
        temp = torch.clone(ic)
        with torch.inference_mode():
            self.nn_model.full_input[0, 4::, :, :] = temp
            _, temp = self.nn_model.net.forward_multi_step(self.nn_model.full_input, temp, self.nn_model.lead_times,
                                                           self.nn_model.hold_variables,
                                                           self.nn_model.hold_out_variables,
                                                           steps=self.NUM_MODEL_STEPS)
        return torch.clone(temp)

    def fourDvar(self, forecast = False):
        self.x = torch.clone(self.background)
        self.x.requires_grad_(True)
        for itr, (self.all_obs, self.H_idxs, self.H_obs, self.n_obs) in enumerate(self.obs_dataloader):
            self.optim = self.configure_optimizer()
            _, cycle_loss = self.cycle(itr, forecast)
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


