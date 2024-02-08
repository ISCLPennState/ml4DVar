import os, sys
#sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")

sys.path.append("/eagle/MDClimSim/mjp5595/ClimaX-v2/src")
import torch
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
from src.dv import *
from src.obs import *

from torch.utils.data import IterableDataset, DataLoader, Dataset
from torchvision.transforms import transforms
from pytorch_lightning import LightningDataModule

# Local application

from climax.global_forecast.iterative_dataset import ERA5OneStepDataset, ERA5MultiLeadtimeDataset

from climax.arch import ClimaX

def calc_background_err(x,background,background_err,num_vars,dv_layer,sht,sht_scaler,print_loss,save_loss_comps,device):
    # Compute background error with identity background error covariance
    dvx = dv_layer(x).to(device)
    dvb = dv_layer(background).to(device)
    diff = dvx - dvb
    sht_diff = sht(diff)

    se_background_comps_unscaled = torch.abs(sht_diff.to(device) * torch.conj(sht_diff.to(device)))

    # TODO check sht_scaler
    se_background_comps = torch.sum(se_background_comps_unscaled / torch.unsqueeze(background_err.to(device), 2) * sht_scaler, (1,2))
    se_background = torch.sum(se_background_comps)
    if print_loss:
        print('\tBackground :',se_background.item())
    save_array = np.zeros((0))
    if save_loss_comps:
        save_array = np.zeros(3*num_vars)
        save_array[:num_vars] = se_background_comps.detach().cpu().numpy()
    return se_background, diff, sht_diff, save_array

def calc_hf_err(diff,sht_diff,num_vars,inv_sht,background_err_hf,print_loss,save_array,device):
    hf_diff = diff - inv_sht(sht_diff)
    se_background_hf_comps = torch.sum(torch.abs(hf_diff)**2.0 / background_err_hf, (1,2))
    se_background_hf = torch.sum(se_background_hf_comps)
    if print_loss:
        print('\tBackground HF:',se_background_hf.item())
    if len(save_array) > 1:
        save_array[num_vars:2*num_vars] = se_background_hf_comps.detach().cpu().numpy()
    return se_background_hf, save_array

def calc_obs_err0(vars,num_vars,x,H_idxs,num_obs,H_obs,obs_err,obs,print_loss,save_array,device,step=0):
    #Compute error in observations at first time step for all variables
    se_obs = torch.zeros(1).to(device)
    for var in range(num_vars):
        x_obs = observe_linear(x[var].reshape(-1, 1),
                               H_idxs[step, var, :4*num_obs[step, var]].reshape(-1, 4).T,
                               H_obs[step, var, :4*num_obs[step, var]].reshape(-1, 4))
        var_err = obs_err(x_obs, obs[step, var, :num_obs[step, var]], var)
        if torch.isnan(var_err):
            var_err = 0
        if print_loss:
            print('\t{}: {}'.format(vars[var],var_err))
        if len(save_array) > 1:
            save_array[var+(2*num_vars)] = var_err.detach().cpu().numpy()
        se_obs += var_err
    if print_loss:
        print('\tOBS (step{}): {}'.format(step,se_obs.item()))
    return se_obs, save_array

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
    obs = args[3]
    H_idxs = args[4]
    H_obs = args[5]
    num_obs = args[6]
    obs_err = args[7]
    #nn_model = args[8] # not used
    climaX_Wrapper = args[8] # not used
    dv_layer = args[9]
    sht = args[10]
    inv_sht = args[11]
    sht_scaler = args[12]
    print_loss = args[13]
    save_loss_comps = args[14]

    vars = args[15]
    device = args[16]

    obs = obs.to(device)

    num_vars = len(vars)
    time_steps = obs.shape[0]
    #if print_loss:
    #    print('In threeDresidual')
    #    print('\ttime_steps:',time_steps)
    #    #print('\t(threeDresidual) vars:',vars)

    se_background, diff, sht_diff, save_array = calc_background_err(x,background,background_err,
                                                                      num_vars,
                                                                      dv_layer,sht,sht_scaler,
                                                                      print_loss,save_loss_comps,
                                                                      device,)

    se_background_hf, save_array = calc_hf_err(diff,sht_diff,num_vars,inv_sht,
                                               background_err_hf,
                                               print_loss,save_array,
                                               device,)

    se_obs, save_array = calc_obs_err0(vars,num_vars,x,
                                       H_idxs,num_obs,H_obs,obs_err,obs,
                                       print_loss,save_array,
                                       device,step=0)
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
    background_err_hf = args[2]
    obs = args[3]
    H_idxs = args[4]
    H_obs = args[5]
    num_obs = args[6]
    obs_err = args[7]
    #nn_model = args[7]
    climaX_Wrapper = args[8]
    dv_layer = args[9]
    sht = args[10]
    inv_sht = args[11]
    sht_scaler = args[12]
    print_loss = args[13]
    save_loss_comps = args[14]

    vars = args[15]
    NUM_MODEL_STEPS = args[16]
    device = args[17]

    obs = obs.to(device)

    num_vars = x.shape[0]
    time_steps = obs.shape[0]

    #if print_loss:
    #    print('In fourDresidual')
    #    print('\ttime_steps:',time_steps)

###################################################################################################################################
    se_background, diff, sht_diff, save_array = calc_background_err(x,background,background_err,
                                                                      num_vars,
                                                                      dv_layer,sht,sht_scaler,
                                                                      print_loss,save_loss_comps,
                                                                      device,)

    se_background_hf, save_array = calc_hf_err(diff,sht_diff,num_vars,inv_sht,
                                               background_err_hf,
                                               print_loss,save_array,
                                               device,)

    se_obs0, save_array = calc_obs_err0(vars,num_vars,x,
                                       H_idxs,num_obs,H_obs,obs_err,obs,
                                       print_loss,save_array,
                                       device,step=0)

    # Compute effects on later observations
    #se_obs_2 = torch.zeros(1).to(device)
    #for step in range(1, time_steps):
    #    # Update model state using forward model
    #    #temp = torch.clone(x)
    #    #with torch.inference_mode():
    #    print('(fourDresidual) x.size:',x.size)
    #    temp = torch.clone(x.to(device))
    #    with torch.inference_mode():
    #        nn_model.full_input = torch.cat((nn_model.full_input[:,:4,:,:],torch.unsqueeze(temp,dim=0)),dim=1)
    #        _, x = nn_model.net.forward_multi_step(nn_model.full_input, temp, nn_model.lead_times, nn_model.hold_variables, nn_model.hold_out_variables,steps=NUM_MODEL_STEPS) 
    #        #x = nn_model.forward(x) 
    #        # #_, x = nn_model.net.forward_multi_step(nn_model.full_input, x, nn_model.lead_times, nn_model.hold_variables, nn_model.hold_out_variables,steps=NUM_MODEL_STEPS)

###################################################################################################################################
###################################################################################################################################
    #x2 = torch.clone(x.to(device))
    #print('(fourDresidual) (0) x.requires_grad',x.requires_grad)
    #print('(fourDresidual) (0) x2.requires_grad',x2.requires_grad)
    for step in range(1, time_steps):
        #with torch.inference_mode():
        norm_preds,_,_ = climaX_Wrapper.forward_multi_step(x.unsqueeze(0).to(device),vars,NUM_MODEL_STEPS)
        #norm_preds = torch.stack(norm_preds, dim=1).flatten(0, 1)
        #print('(fourDresidual) norm_preds step :',len(norm_preds),step)
        x2 = norm_preds[-1]
    #print('(fourDresidual) norm_preds done :',len(norm_preds))
    #x2 = norm_preds
###################################################################################################################################
###################################################################################################################################

    print('(fourDresidual) (0) x2.requires_grad',x2.requires_grad)
    print('(fourDresidual) (0) climaX_Wrapper.net.requires_grad',climaX_Wrapper.net.requires_grad)
    se_obs1, save_array = calc_obs_err0(vars,num_vars,x2.squeeze(0),
                                       H_idxs,num_obs,H_obs,obs_err,obs,
                                       print_loss,save_array,
                                       device,step=1)

    #print('(fourDresidual) se_background :',se_background)
    #print('(fourDresidual) se_background_hf :',se_background_hf)
    #print('(fourDresidual) se_obs0 :',se_obs0)
    print('(fourDresidual) se_obs1 :',se_obs1)

    if save_loss_comps:
        return se_background + se_background_hf + se_obs0 + se_obs1, save_array
    else:
        return se_background + se_background_hf + se_obs0 + se_obs1

class FourDVar():
    def __init__(self, climaX_Wrapper, obs_dataloader, background, background_err, background_err_hf, obs_err, dv_layer,
                 lr = 1., max_iter = 700, forecast_steps = 20, spin_up_cycles = 9, runstr = None, save_analysis = True,
                 savedir = None, NUM_MODEL_STEPS=2, vars=None, device=None, save_idx=0):
        super(FourDVar).__init__()
        self.save_hyperparameters()
        self.sht = th.RealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.inv_sht = th.InverseRealSHT(background.shape[2], background.shape[3], grid="equiangular").to(device).float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2)).reshape(1, 1, -1).to(device)
        self.save_dir = savedir

        self.climaX_Wrapper = climaX_Wrapper

        if not self.save_dir:
            self.save_dir = os.path.join(os.getcwd(), 'data')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        if not self.runstr:
            self.runstr = "%dhr_%s" % (obs_dataloader.dataset.window_step, obs_dataloader.dataset.start_datetime.strftime("%m%d%Y"))
        self.vars = vars
        self.optim = None
        self.NUM_MODEL_STEPS = NUM_MODEL_STEPS

        self.device = device

        self.save_idx = save_idx 

        debug = True
        if debug:
            print('(FourDVar) Device :',device)
            print('self.vars :',self.vars)
        
    def loss(self, print_loss = False, save_loss_comps = False, itr = 0):
        #if print_loss:
        #    print('(FourDVar.loss) H_idxs.size:',self.H_idxs.size())
        #    print('(fourDVar.loss) self.all_obs',self.all_obs.shape)
        #    print('(fourDVar.loss) self.H_idxs',self.H_idxs.shape)
        #    print('(fourDVar.loss) self.H_obs',self.H_obs.shape)
        #    print('(fourDVar.loss) self.n_obs',self.n_obs.shape)

        if self.H_idxs.size(dim = 1) == 1:
            out = threeDresidual(self.x_analysis[0].to(self.device),
                             self.background[0].to(self.device),
                             self.background_err.to(self.device),
                             self.background_err_hf.to(self.device),
                             self.all_obs[0],
                             self.H_idxs[0],
                             self.H_obs[0],
                             self.n_obs[0],
                             self.obs_err,
                             self.climaX_Wrapper, # not used
                             self.dv_layer,
                             self.sht,
                             self.inv_sht,
                             self.sht_scaler,
                             print_loss,
                             save_loss_comps,
                             self.vars,
                             self.device,
                             )
            if save_loss_comps:
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d_%s.npy' % (itr, self.step, self.runstr))
                np.save(filename, out[1])
                return out[0]
            else:
                return out
        else:
            out = fourDresidual(self.x_analysis[0].to(self.device),
                                 self.background[0].to(self.device),
                                 self.background_err.to(self.device),
                                 self.background_err_hf.to(self.device),
                                 self.all_obs[0],
                                 self.H_idxs[0],
                                 self.H_obs[0],
                                 self.n_obs[0],
                                 self.obs_err,
                                 #self.nn_model,
                                 self.climaX_Wrapper,
                                 self.dv_layer,
                                 self.sht,
                                 self.inv_sht,
                                 self.sht_scaler,
                                 print_loss,
                                 save_loss_comps,
                                 self.vars,
                                 self.NUM_MODEL_STEPS,
                                 self.device,
                                 )
            if save_loss_comps:
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d_%s.npy' % (itr, self.step, self.runstr))
                np.save(filename, out[1])
                return out[0]
            else:
                return out

    def cycle(self, itr, forecast):
        #debug = True
        debug = False

        self.step = 0
        def cost_J():
            #self.nn_model.zero_grad() # for 3D, self.nn_model isn't used
            loss = self.loss(print_loss = False, save_loss_comps = False, itr = itr)
            #loss = self.loss(print_loss = True, save_loss_comps = False, itr = itr)
            #print('self.optim.param_groups :',self.optim.param_groups)
            #print('num self.optim.param_groups :',len(self.optim.param_groups))
            #print('num params[0] self.optim.param_groups :',len(self.optim.param_groups[0]['params']))
            #print('self.optim.param_groups[0] :',self.optim.param_groups[0]['params'][0].shape)
            print('(cost_J) step: {}\tloss: {}'.format(self.step,loss.item()))
            self.step += 1
            loss.backward(retain_graph=False)
            #loss.backward(retain_graph=True)
            self.optim.zero_grad()
            return loss

        if self.save_analysis:
            np.save(os.path.join(self.save_dir, 'background_%d_%s.npy' % (itr, self.runstr)),
                    self.background.detach().cpu().numpy())

        if debug:
            print('(cycle) before optimizing')
        self.optim.step(cost_J)
        if debug:
            print('(cycle) after optimizing')
        if self.save_analysis:
            save_analysis = self.x_analysis.detach().cpu().numpy()
            np.save(os.path.join(self.save_dir, 'analysis_%d_%s.npy' % (itr+self.save_idx, self.runstr)), save_analysis)

        #print('(cycle) x_analysis.shape :',self.x_analysis.shape)

        cycle_loss = self.loss(print_loss = True, save_loss_comps = False)
        if debug:
            print('(cycle) cycle_loss :',cycle_loss.item())

        if debug:
            print('(cycle) running self.run_forecast')
        self.background = self.run_forecast(self.x_analysis)
        if debug:
            print('(cycle) done with self.run_forecast')

        self.x_analysis = torch.clone(self.background)
        self.x_analysis.requires_grad_(True)
        if forecast and itr > self.spin_up_cycles:
            if debug:
                print('(cycle) running self.run_forecasts')
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

    def run_forecast(self, x):
        temp = torch.clone(x)
        with torch.inference_mode():
            norm_preds,raw_preds,norm_diff = self.climaX_Wrapper.forward_multi_step(temp[-1].unsqueeze(0).to(self.device),self.vars,self.NUM_MODEL_STEPS)

        #norm_preds = torch.stack(norm_preds, dim=1).flatten(0, 1)
        norm_preds = norm_preds[-1]
        return torch.clone(norm_preds)

    def configure_optimizer(self):
        #return torch.optim.LBFGS([self.x_analysis], lr = self.lr, max_iter = self.max_iter, history_size=300, tolerance_grad = 1e-5)
        return torch.optim.LBFGS([self.x_analysis], lr = self.lr, max_iter = 20, history_size=300, tolerance_grad = 1e-5)

    def fourDvar(self, forecast = False):
        self.x_analysis = torch.clone(self.background)
        self.x_analysis.requires_grad_(True)
        #print('x_analysis.shape :',self.x_analysis.shape)
        for itr, (all_obs, H_idxs, H_obs, shapes, obs_latlon) in enumerate(self.obs_dataloader):
            self.all_obs = all_obs
            self.H_idxs = H_idxs
            self.H_obs = H_obs
            self.shapes = shapes
            self.obs_latlon = obs_latlon
            self.n_obs = self.shapes
            self.optim = self.configure_optimizer()
            _, cycle_loss = self.cycle(itr, forecast)
            print('Cycle loss %d: %0.2f' % (itr, cycle_loss))
            print('{} / {}'.format(itr,self.obs_dataloader.dataset.num_cycles))

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)