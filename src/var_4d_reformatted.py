import os, sys

import torch
import inspect
import h5py
import numpy as np
import torch_harmonics as th
from src.dv import *
from src.obs_cummulative import *

import time
import pickle as p

torch.autograd.set_detect_anomaly(True)

class FourDVar():
    def __init__(self, stormer_wrapper, obs_dataloader, 
                 background, background_err_dict, background_err_hf_dict, obs_err, wind_layer,
                 wind_type='scalar', model_step=6, da_window=12, obs_freq=3, da_type='var4d', vars=None,
                 b_inflation=1, b_hf_inflation=1, lr=1., max_iter=700, forecast_steps=40, savestr=None,
                 save_analysis=True, savedir=None, device=None, save_idx=0, logger=None,
                 ):
        super(FourDVar).__init__()
        self.save_hyperparameters()

        self.stormer_wrapper = stormer_wrapper
        self.obs_dataloader = obs_dataloader

        self.device = device
        self.background = background.to(self.device)
        self.background_err_dict = background_err_dict
        self.background_err_hf_dict = background_err_hf_dict
        self.obs_err = obs_err 
        self.wind_layer = wind_layer.to(self.device)
        self.wind_type = wind_type

        for hr_key in self.background_err_dict.keys():
            self.background_err_dict[hr_key] = self.background_err_dict[hr_key] * b_inflation
            self.background_err_hf_dict[hr_key] = self.background_err_hf_dict[hr_key] * b_hf_inflation
        self.se_obs = torch.zeros(1).to(self.device)

        self.model_step = model_step
        self.da_window = da_window
        self.obs_freq = obs_freq
        self.da_type = da_type
        self.vars = vars
        self.forecast_steps = forecast_steps
        if self.da_type not in ['var3d','var4d']:
            print('da_type must be in ["var3d","var4d"]')

        self.save_idx = save_idx 
        self.logger = logger

        self.save_dir = savedir
        if not self.save_dir:
            self.save_dir = os.path.join(os.getcwd(), 'data')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        if not self.savestr:
            self.savestr = "%s" % (obs_dataloader.dataset.curr_datetime.strftime("%H%d%m%Y"))
        self.optim = None
        
    def loss(self, print_loss = False, save_loss_comps = False, itr = 0):

        if self.H_idxs.size(dim = 1) == 1:
            out = self.threeDresidual(print_loss, save_loss_comps)
            if save_loss_comps:
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d_%s.npy' % (itr, self.step, self.savestr))
                np.save(filename, out[1])
                return out[0]
            else:
                return out
        else:
            out = self.fourDresidual(print_loss, save_loss_comps)
            if save_loss_comps:
                filename = os.path.join(self.save_dir, 'loss_comps_cycle%d_step%d_%s.npy' % (itr, self.step, self.savestr))
                np.save(filename, out[1])
                return out[0]
            else:
                return out

    def cycle(self, itr, forecast):

        if self.save_analysis:
            np.save(os.path.join(self.save_dir, 'background_%d_%s.npy' % (itr+self.save_idx, self.savestr)),
                    self.background.detach().cpu().numpy())

        self.step = 0
        def cost_J():
            self.optim.zero_grad()
            self.stormer_wrapper.net.zero_grad()
            loss = self.loss(print_loss = False, save_loss_comps = False, itr = itr)
            print('(cost_J) step: {}\tloss: {}'.format(self.step,loss.item()))
            #print('mem_info :',torch.cuda.mem_get_info())

            if self.logger:
                self.logger.info('(cost_J) step: {}\tloss: {}'.format(self.step,loss.item()))
            self.step += 1
            loss.backward(retain_graph=False)

            return loss

        ############################################################################
        # Set the background_err to use for each cycle
        ############################################################################
        bg_hr_key = int(self.savestr[:2])
        if self.logger:
            self.logger.info('self.savestr {}'.format(self.savestr))
            self.logger.info('Using background_err_dict[{}]'.format(bg_hr_key))
        #bg_hr_key = int(self.savestr[:2])
        self.background_err = self.background_err_dict[bg_hr_key]
        self.background_err_hf = self.background_err_hf_dict[bg_hr_key]
        ############################################################################
        ############################################################################

        # This optimizes x_analysis given the observations (data assimilation occurs) x -> x_analysis
        tic = time.perf_counter()
        self.optim.step(cost_J)
        toc = time.perf_counter()
        print('Cycling took {:0.4f} seconds'.format(toc-tic))
        if self.logger:
            self.logger.info('Cycling took {:0.4f} seconds'.format(toc-tic))

        cycle_loss = self.loss(print_loss=True, save_loss_comps=False)
        cycle_loss.backward(retain_graph=False)
        self.optim.zero_grad()
        self.stormer_wrapper.net.zero_grad()

        # This brings x_analysis to the end of the DA window
        if self.da_type == 'var4d':    
            x_analysis_list,_,_ = self.run_forecast(self.x_analysis,
                                                    forecast_time=self.da_window-self.model_step,
                                                    lead_time=self.model_step,
                                                    inference=True)
            self.x_analysis = x_analysis_list[-1].unsqueeze(0)

        if self.save_analysis:
            save_analysis = self.x_analysis.detach().cpu().numpy()
            np.save(os.path.join(self.save_dir, 'analysis_%d_%s.npy' % (itr+self.save_idx, self.savestr)), save_analysis)

        # gets and saves forecasts first before reassigning any variables
        if forecast:
            print('Saving forecasts')
            if self.logger:
                self.logger.info('Saving forecasts')
                self.save_forecasts(itr)

        # Get new background (model forecast from current optimized x_analysis)
        # for 3dvar this should be 12hrs (2 model steps)
        # for 4dvar this should be 6hrs (1 model step) - 4dvar should always be just min(obs_window,model_step)
        if self.da_type == 'var3d':
            forecast_time = self.da_window
        if self.da_type == 'var4d':
            forecast_time = self.model_step
        #with torch.inference_mode():
        print('Running forecast for next background (forecast_time : {})'.format(forecast_time))
        if self.logger:
            self.logger.info('Running forecast for next background (forecast_time : {} hrs)'.format(forecast_time))
        bg_list,_,_ = self.run_forecast(self.x_analysis,
                                                forecast_time=forecast_time,
                                                lead_time=None,
                                                #lead_time=self.model_step,
                                                inference=True)
        self.background = bg_list[-1].unsqueeze(0)
        # Background is the forecast of previous analysis

        # x_analysis becomes forecast of previous analysis
        print('Setting new x_analysis to new background')
        if self.logger:
            self.logger.info('Setting new x_analysis to new background')
        self.x_analysis = torch.clone(self.background).detach()
        self.x_analysis.requires_grad_(True)

        self.savestr = "%s" % (self.obs_dataloader.dataset.curr_datetime.strftime("%H%d%m%Y"))
        return self.background, cycle_loss

    def fourDresidual(self, print_loss, save_loss_comps):
        # x_analysis is forecast (1step) from prev x_analysis / curr_background
        # x_analysis == background @ t_0/6hrs
        x = self.x_analysis[0]

        # t_0 -> 6hrs
        # t_1 -> 12hrs
        se_background, diff, sht_diff, save_array = self.calc_background_err(x, print_loss, save_loss_comps)
        se_background_hf, save_array = self.calc_hf_err(diff, sht_diff, print_loss, save_array)
        # calc obs err @ step 0 (6hrs)
        se_obs, save_array = self.calc_obs_err(x, print_loss, save_array, obs_step=0)

        #############################################################################################################
        # get forecast prediction for step1 (6hrs -> 12hrs)
        # prediction based on x (x_analysis)
        #############################################################################################################
        for step in range((self.da_window//self.model_step) - 1):
            if step == 0:
                x_temp,_,_ = self.run_forecast(x.unsqueeze(0),
                                               forecast_time=self.model_step,
                                               lead_time=self.model_step,
                                               print_steps=False,)
            else:
                x_temp,_,_ = self.run_forecast(x_temp[0].unsqueeze(0),
                                               forecast_time=self.model_step,
                                               lead_time=self.model_step,
                                               print_steps=False,)
            se_obs_temp, save_array = self.calc_obs_err(x_temp[0], print_loss, save_array, obs_step=step+1)
            se_obs += se_obs_temp

        #print('(fourDresidual) se_background :',se_background)
        #print('(fourDresidual) se_background_hf :',se_background_hf)
        #print('(fourDresidual) se_obs:',se_obs)

        if save_loss_comps:
            return 0.5*se_background + se_background_hf + 0.5*(se_obs), save_array
        else:
            return 0.5*se_background + se_background_hf + 0.5*(se_obs)

    def run_forecast(self, x, forecast_time, lead_time=None, inference=False, print_steps=True):
        # norm_preds: [(num_vars,lat,lon)]*num_steps -> [(63,128,256)]*num_steps
        if inference:
            with torch.inference_mode():
                norm_preds, raw_preds, lead_time_combos = self.stormer_wrapper.eval_to_forecast_time_with_lead_time(
                    x,
                    forecast_time=forecast_time,
                    lead_time=lead_time,
                    print_steps=print_steps,
                    )
        else:
            norm_preds, raw_preds, lead_time_combos = self.stormer_wrapper.eval_to_forecast_time_with_lead_time(
                x,
                forecast_time=forecast_time,
                lead_time=lead_time,
                print_steps=print_steps,
                )
        return norm_preds, raw_preds, lead_time_combos

    def save_forecasts(self, itr):
        temp = torch.clone(self.x_analysis).detach()
        norm_forecasts,raw_forecasts,lead_time_combo = self.run_forecast(temp,
                                                                          forecast_time=self.forecast_steps*self.model_step,
                                                                          lead_time=self.model_step,
                                                                          inference=True)

        hf_norm = h5py.File(os.path.join(self.save_dir, 'forecasts_%d_%s.h5' % (itr+self.save_idx, self.savestr)),'w')
        hf_raw = h5py.File(os.path.join(self.save_dir, 'raw_forecasts_%d_%s.h5' % (itr+self.save_idx, self.savestr)),'w')
        hf_norm.create_dataset(str(0), data=self.x_analysis[0].detach().cpu().numpy())
        #hf_raw.create_dataset(str(0), data=self.x_analysis[0].detach().cpu().numpy())
        hf_raw.create_dataset(str(0), data=self.stormer_wrapper.reverse_inp_transform(self.x_analysis)[0].detach().cpu().numpy())
        forecast_time = 0
        for i in range(len(norm_forecasts)):
            hf_norm.create_dataset(str(forecast_time+lead_time_combo[i]), data=norm_forecasts[i].detach().cpu().numpy())
            hf_raw.create_dataset(str(forecast_time+lead_time_combo[i]), data=raw_forecasts[i].detach().cpu().numpy())
            forecast_time += lead_time_combo[i]
        return

    def configure_optimizer(self):
        return torch.optim.LBFGS([self.x_analysis], lr = self.lr, max_iter = self.max_iter, history_size=300, tolerance_grad = 1e-5)
        #return torch.optim.LBFGS([self.x_analysis], lr = self.lr, max_iter = 200, history_size=300, tolerance_grad = 1e-5)
        #return torch.optim.LBFGS([self.x_analysis], lr = self.lr, max_iter = 10, history_size=300, tolerance_grad = 1e-5)

    def cycleDataAssimilation(self, forecast = False):
        
        self.x_analysis = torch.clone(self.background)
        self.x_analysis.requires_grad_(True)

        for itr, (self.all_obs, self.H_idxs, self.H_obs, self.shapes, self.obs_latlon) in enumerate(self.obs_dataloader):
            self.n_obs = self.shapes
            self.optim = self.configure_optimizer()
            _, cycle_loss = self.cycle(itr, forecast)
            self.optim.zero_grad()
            print('Cycle loss %d: %0.2f' % (itr+self.save_idx, cycle_loss))
            print('{} / {}'.format(itr+self.save_idx,self.obs_dataloader.dataset.num_cycles))
            #print('{} / {}'.format(itr,len(self.obs_dataloader)))
            if self.logger:
                self.logger.info('Cycle loss {:d}: {:0.2f}'.format(itr+self.save_idx, cycle_loss.item()))
                self.logger.info('{} / {}'.format(itr+self.save_idx,self.obs_dataloader.dataset.num_cycles))
                #self.logger.info('{} / {}'.format(itr,len(self.obs_dataloader)))

    def calc_background_err(self, x, print_loss, save_loss_comps):
        # Compute background error  
        dvx = self.wind_layer(x)
        dvb = self.wind_layer(self.background[0])
        diff = dvx - dvb
        sht_diff = self.wind_layer.diffSHT(diff)
        #if self.logger:
        #    self.logger.info('dvx.shape min/max : {},{},{}'.format(dvx.shape,torch.min(dvx),torch.max(dvx)))
        #    self.logger.info('dvb.shape min/max : {},{},{}'.format(dvb.shape,torch.min(dvb),torch.max(dvb)))
        #    self.logger.info('diff.shape min/max/nan/complex : {},{},{},{},{}'.format(diff.shape,
        #                                                                              torch.min(diff),
        #                                                                              torch.max(diff),
        #                                                                              torch.isnan(diff).any(),
        #                                                                              torch.is_complex(diff)
        #                                                                              ))
        #    self.logger.info('sht_diff.shape min/max/nan/complex : {},{},{},{},{}'.format(sht_diff.shape,
        #                                                                   torch.min(torch.real(sht_diff)),
        #                                                                   torch.max(torch.real(sht_diff)),
        #                                                                   torch.isnan(sht_diff).any(),
        #                                                                   torch.is_complex(sht_diff)
        #                                                                   ))

        #se_background_comps_unscaled = torch.abs(sht_diff.to(self.device) * torch.conj(sht_diff.to(self.device)))
        #se_background_comps_unscaled = sht_diff.to(self.device) * torch.conj(sht_diff.to(self.device))
        se_background_comps_unscaled = sht_diff * torch.conj(sht_diff)
        #se_background_comps_unscaled = torch.real(se_background_comps_unscaled)
        se_background_comps_unscaled = torch.real(torch.abs(se_background_comps_unscaled))
        #if self.logger:
        #    self.logger.info('se_background_comps_unscaled.shape min/max/nan/complex : {},{},{},{},{}'.format(se_background_comps_unscaled.shape,
        #                                                                                       torch.min(torch.real(se_background_comps_unscaled)),
        #                                                                                       torch.max(torch.real(se_background_comps_unscaled)),
        #                                                                                       torch.isnan(se_background_comps_unscaled).any(),
        #                                                                                       torch.is_complex(se_background_comps_unscaled)
        #                                                                                       ))
        #if self.logger:
        #    self.logger.info('background_err.shape min/max/nan/complex : {},{},{},{},{}'.format(self.background_err.shape,
        #                                                                                       torch.min(torch.real(self.background_err)),
        #                                                                                       torch.max(torch.real(self.background_err)),
        #                                                                                       torch.isnan(self.background_err).any(),
        #                                                                                       torch.is_complex(self.background_err)
        #                                                                                       ))

        # TODO check sht_scaler
        #se_background_comps = torch.sum(se_background_comps_unscaled / torch.unsqueeze(self.background_err.to(self.device), 2) * self.wind_layer.sht_scaler, (1,2))
        divisor = torch.unsqueeze(self.background_err, 2)
        divisor_nonzero = divisor.clone()
        divisor_nonzero[divisor_nonzero==0] = 1
        se_background_comps_overB = se_background_comps_unscaled / divisor_nonzero
        divisor_broadcasted = divisor.expand(-1,-1,se_background_comps_overB.shape[-1])
        se_background_comps_overB[divisor_broadcasted==0] = 0

        #if self.logger:
        #    self.logger.info('se_background_comps_overB.shape min/max/nan/complex : {},{},{},{},{}'.format(se_background_comps_overB.shape,
        #                                                                   torch.min(torch.real(se_background_comps_overB)),
        #                                                                   torch.max(torch.real(se_background_comps_overB)),
        #                                                                   torch.isnan(se_background_comps_overB).any(),
        #                                                                   torch.is_complex(se_background_comps_overB)
        #                                                                   ))
        ##if self.wind_type == 'vector':
        ##    se_background_comps_overB = torch.nan_to_num(se_background_comps_overB, nan=0)
        ##    #num_tot_wind_idxs = self.wind_layer.uwind_num + self.wind_layer.vwind_num
        ##    #se_background_comps_overB[-num_tot_wind_idxs:,0,:] = 0

        #if self.logger:
        #    self.logger.info('se_background_comps_overB.shape min/max/nan/complex : {},{},{},{},{}'.format(se_background_comps_overB.shape,
        #                                                                   torch.min(torch.real(se_background_comps_overB)),
        #                                                                   torch.max(torch.real(se_background_comps_overB)),
        #                                                                   torch.isnan(se_background_comps_overB).any(),
        #                                                                   torch.is_complex(se_background_comps_overB)
        #                                                                   ))
        se_background_comps_scaled = se_background_comps_overB * self.wind_layer.sht_scaler
        se_background_comps = torch.sum(se_background_comps_scaled,(1,2))
        #if self.logger:
        #    self.logger.info('se_background_comps.shape min/max/nan/complex : {},{},{},{},{}'.format(se_background_comps.shape,
        #                                                                   torch.min(torch.real(se_background_comps)),
        #                                                                   torch.max(torch.real(se_background_comps)),
        #                                                                   torch.isnan(se_background_comps).any(),
        #                                                                   torch.is_complex(se_background_comps)
        #                                                                   ))

        se_background = torch.sum(se_background_comps)
        if print_loss:
            print('\tBackground :',se_background.item())
            if self.logger:
                self.logger.info('\tBackground : {}'.format(se_background.item()))

        save_array = np.zeros((0))
        if save_loss_comps:
            save_array = np.zeros(3*len(self.vars))
            save_array[:len(self.vars)] = se_background_comps.detach().cpu().numpy()
        return se_background, diff, sht_diff, save_array

    def calc_hf_err(self, diff, sht_diff, print_loss, save_array):
        inv_sht = self.wind_layer.diffInvSHT(sht_diff)
        hf_diff = diff - inv_sht
        #se_background_hf_comps = torch.sum(torch.abs(hf_diff)**2.0 / self.background_err_hf.to(self.device), (1,2))
        se_background_hf_comps = torch.sum(torch.abs(hf_diff)**2.0 / self.background_err_hf, (1,2))
        se_background_hf = torch.sum(se_background_hf_comps)

        #if self.logger:
        #    self.logger.info('inv_sht.shape min/max : {},{},{},{}'.format(inv_sht.shape,torch.min(inv_sht),torch.max(inv_sht),torch.is_complex(inv_sht)))
        #    self.logger.info('hf_diff.shape min/max : {},{},{},{}'.format(hf_diff.shape,torch.min(hf_diff),torch.max(hf_diff),torch.is_complex(hf_diff)))
        #    self.logger.info('se_background_hf_comps.shape min/max : {},{},{},{}'.format(se_background_hf_comps.shape,torch.min(se_background_hf_comps),torch.max(se_background_hf_comps),torch.is_complex(se_background_hf_comps)))
        #    self.logger.info('se_background_hf.shape min/max : {},{},{},{}'.format(se_background_hf.shape,torch.min(se_background_hf),torch.max(se_background_hf),torch.is_complex(se_background_hf)))

        if print_loss:
            print('\tBackground HF:',se_background_hf.item())
            if self.logger:
                self.logger.info('\tBackground HF: {}'.format(se_background_hf.item()))

        if len(save_array) > 1:
            save_array[len(self.vars):2*len(self.vars)] = se_background_hf_comps.detach().cpu().numpy()
        return se_background_hf, save_array

    def calc_obs_err(self, x, print_loss, save_array, obs_step=0):
        #Compute error in observations at time step (step) for all variables

        # step0 -> 12hrs for 3dvar
        #
        # step0 -> 6hrs for 4dvar
        # step1 -> 12hrs for 4dvar
        #
        # Obs are (b, time_step, vars, num_obs)

        #se_obs = torch.zeros(1).to(self.device)
        self.se_obs = 0
        for var in range(len(self.vars)):
            # x[var] - (128, 256)
            # H_idx - (1, obs_steps, num_vars, 4*num_obs)
            # H_obs - (1, obs_steps, num_vars, 4*num_obs)
            # all_obs - (1, obs_steps, num_vars, num_obs)

            x_obs = observe_linear(x[var].reshape(-1, 1),
                                self.H_idxs[0, obs_step, var, :4*self.n_obs[0,  obs_step, var]].reshape(-1, 4).T,
                                self.H_obs[0, obs_step, var, :4*self.n_obs[0, obs_step, var]].reshape(-1, 4),
                                #self.logger
                                )
            var_err = self.obs_err(x_obs, self.all_obs[0, obs_step, var, :self.n_obs[0, obs_step, var]], var)

            if torch.isnan(var_err):
                var_err = 0
            if print_loss:
                print('\t{}: {}'.format(self.vars[var],var_err))
                if self.logger:
                    self.logger.info('\t{}: {}'.format(self.vars[var],var_err))

            if len(save_array) > 1:
                save_array[var+(2*len(self.vars))] = var_err.detach().cpu().numpy()
            self.se_obs += var_err
        if print_loss:
            print('\tOBS (step{}): {}'.format(obs_step,self.se_obs.item()))
            if self.logger:
                self.logger.info('\tOBS (step{}): {}'.format(obs_step,self.se_obs.item()))

        return self.se_obs, save_array

    def threeDresidual(self, print_loss, save_loss_comps):

        x = self.x_analysis[0]

        se_background, diff, sht_diff, save_array = self.calc_background_err(x, print_loss, save_loss_comps)
        se_background_hf, save_array = self.calc_hf_err(diff, sht_diff, print_loss, save_array)
        se_obs, save_array = self.calc_obs_err(x, print_loss, save_array, obs_step=0)

        #if self.logger:
        #    self.logger.info('se_background val,complex : {},{}'.format(se_background,torch.is_complex(se_background)))
        #    self.logger.info('se_background_hf val,complex : {},{}'.format(se_background_hf,torch.is_complex(se_background_hf)))
        #    self.logger.info('se_obs val,complex : {},{}'.format(se_obs.shape,torch.is_complex(se_obs)))

        if save_loss_comps:
            return se_background + se_background_hf + se_obs, save_array
        else:
            return 0.5*se_background + se_background_hf + 0.5*se_obs

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)