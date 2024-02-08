import os, sys
#sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from torch.utils.data import IterableDataset, DataLoader
import torch
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
from src.dv import *

class ObsError(torch.nn.Module):
    def __init__(self, vars, var_types, var_obs_errs, obs_perc_errs, var_stds):
        super().__init__()
        print('vars :',vars)
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

    def forward(self, obs, x_obs, var, logger=None):
        if logger:
            logger.info('\tObsError obs : {}'.format(obs))
            logger.info('\tObsError self.obs_err[var] : {}'.format(self.obs_err[var]))
            logger.info('\tObsError self.mult_vars[var] : {}'.format(self.mult_vars[var]))
        if self.mult_vars[var]:
            return torch.sum((x_obs - obs)**2.0 / (self.obs_err[var] * torch.abs(obs)) ** 2.0)
        else:
            return torch.sum((x_obs - obs)**2.0) / self.obs_err[var]

def observe_linear(x, H_idxs, H_vals, logger=None):
    # TODO maybe this should be mean, but I don't think it really matters
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1),
                       axis = 1).to(device)
    if logger:
        logger.info('\tx.shape : {}'.format(x.shape))
        logger.info('\tH_idxs : {}'.format(H_idxs))
        logger.info('\tH_idxs.shape : {}'.format(H_idxs.shape))
        logger.info('\tH_vals : {}'.format(H_vals))
        logger.info('\tH_vals.shape : {}'.format(H_vals.shape))
        logger.info('\toutput : {}'.format(output))
        logger.info('\toutput.shape : {}'.format(output.shape))

    return output

class ObsDatasetCum(IterableDataset):
    def __init__(self, file_path, start_datetime, end_datetime, vars, 
                 obs_freq=3, da_window=12, obs_start_idx=0, obs_steps=1,
                 logger=None):
        super().__init__()
        self.save_hyperparameters()
        self.file_path = file_path

        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        datetime_diff = self.end_datetime - self.start_datetime
        hour_diff = datetime_diff.days*24 + datetime_diff.seconds // 3600
        self.num_cycles = int(hour_diff/obs_freq) - da_window

        self.vars = vars
        self.obs_freq = obs_freq
        self.da_window = da_window 
        self.obs_start_idx = obs_start_idx
        self.obs_steps = obs_steps

        self.start_datetime = self.start_datetime + timedelta(hours=self.da_window*self.obs_start_idx)
        self.curr_datetime = self.start_datetime + timedelta(hours=self.da_window)

    # This accumulates the observations that take place within the obs_window up/including the current datetime
    # eg. for obs_window=12hrs and a datetime of (2014,1,1,12) it will accumulate the datetimes (2014,1,1,3),(2014,1,1,6),(2014,1,1,9),(2014,1,1,12)
    # with a obs_window=6hrs and obs_steps=2 it will accumulate [(2014,1,1,3),(2014,1,1,6)],[(2014,1,1,9),(2014,1,1,12)]
    # The only potential problem is if the same var observation takes place at the latlon, but at a different time. IDK how this will be interpreted later, but for now both would be returned
    def read_file(self):
        with h5py.File(self.file_path, 'r') as f:
            # shapes -> (steps,vars) -> holds max number of observations per var
            shapes = np.zeros((self.obs_steps, len(self.vars)), dtype = int)
            obs_datetimes = [self.start_datetime + timedelta(hours=int((i+1)*self.obs_freq)) for i in range(int(self.da_window/self.obs_freq))]
            obs_per_step = self.da_window // (self.obs_freq*self.obs_steps)

            print('all obs_datetimes :',obs_datetimes)
            if self.logger:
                self.logger.info('')
                self.logger.info('All obs_datetimes : {}'.format(obs_datetimes))
                self.logger.info('Counting Obs at each Datetime')

            total_obs = []
            dt_obs_dict = {}
            for dt in obs_datetimes:
                dt_obs_dict[dt] = 0
            for step in range(self.obs_steps):
                step_obs = 0
                for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes[obs_per_step*step:obs_per_step*(step+1)]), enumerate(self.vars)):
                    try:
                        if var not in f[obs_datetime.strftime("%Y/%m/%d/%H") + '/'].keys():
                            continue
                    except:
                        #if self.logger and j==0:
                        #    self.logger.info('\tobs_datetime not found : {}'.format(obs_datetime))
                        continue
                    #if self.logger and j==0:
                    #    self.logger.info('(0) step : {},\tobs_datetime : {}'.format(step,obs_datetime))
                    #shapes[step, j] = max(shapes[step,j],f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var].shape[0])
                    dt_obs = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var].shape[0]
                    shapes[step, j] += dt_obs
                    step_obs += dt_obs
                    dt_obs_dict[obs_datetime] += dt_obs
                total_obs.append(step_obs)

            max_obs = np.max(shapes)
            all_obs = np.zeros((self.obs_steps, len(self.vars), max_obs))
            H_idxs = np.zeros((self.obs_steps, len(self.vars), 4*max_obs), dtype = 'i4')
            H_obs = np.zeros((self.obs_steps, len(self.vars), 4*max_obs))
            obs_latlon = np.zeros((self.obs_steps, len(self.vars), max_obs, 2))

            print('datetime_obs_dict : {}'.format(dt_obs_dict))
            if self.logger:
                self.logger.info('datetime_obs_dict : {}'.format(dt_obs_dict))

            #obs_per_step = self.da_window // (self.obs_freq*self.obs_steps)
            #for step in range(self.obs_steps):
            #    o_datetimes = obs_datetimes[obs_per_step*step:obs_per_step*(step+1)]
            #    #o_datetimes = obs_datetimes[step]
            #    print('step : {},\tobs_datetimes : {},\tnum_obs : {}'.format(step,o_datetimes,total_obs[step]))
            #    if self.logger:
            #        self.logger.info('step : {},\tobs_datetimes : {},\t num_obs : {}'.format(step,o_datetimes,total_obs[step]))

            print('Gathering Observations')
            if self.logger:
                self.logger.info('Gathering Observations')

            for step in range(self.obs_steps):
                var_starts = np.zeros(len(self.vars),dtype=int)
                for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes[obs_per_step*step:obs_per_step*(step+1)]), enumerate(self.vars)):
                #for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes[step]), enumerate(self.vars)):
                    try:
                        if var not in f[obs_datetime.strftime("%Y/%m/%d/%H") + '/'].keys():
                            continue
                    except:
                        if self.logger and j==0:
                            self.logger.info('\tobs_datetime not found : {}'.format(obs_datetime))
                        continue
                    if self.logger and j==0:
                        self.logger.info('\tobs_step : {},\tobs_datetime : {}'.format(step,obs_datetime))
                    all_obs_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 2]
                    all_obs[step, j, var_starts[j]:var_starts[j]+len(all_obs_data)] = all_obs_data
                    H_idxs_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 0]
                    H_idxs[step, j, 4*var_starts[j]:4*var_starts[j]+len(H_idxs_data)] = H_idxs_data
                    H_obs_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 1]
                    H_obs[step, j, 4*var_starts[j]:4*var_starts[j]+len(H_obs_data)] = H_obs_data
                    #obs_latlon_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, :2]

                    #################################################################################################
                    ## This is the original block, but it expects obs to be 0 -> 360
                    #################################################################################################
                    ## temp fix
                    #obs_lat_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 0:1]
                    #obs_lon_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 1:2]
                    #if len(obs_lon_data) > 0:
                    #    print('min/max(obs_lon_data) :',min(obs_lon_data),max(obs_lon_data))
                    #obs_lon_data = np.where(obs_lon_data > 180.0, obs_lon_data-360.0, obs_lon_data)
                    ##print('obs_lat_data.shape :',obs_lat_data.shape)
                    ##print('obs_lon_data.shape :',obs_lon_data.shape)
                    #obs_latlon_data = np.concatenate((obs_lat_data,obs_lon_data),axis=1)
                    ##print('obs_latlon_data.shape :',obs_latlon_data.shape)
                    #################################################################################################
                    #################################################################################################

                    #################################################################################################
                    ## This is the original block, but it expects obs to be -180 -> 180 
                    ## ERA5 lon is different than IRGA lon
                    #################################################################################################
                    ## temp fix
                    #obs_lat_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 0:1]
                    #obs_lon_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 1:2]
                    #if len(obs_lon_data) > 0:
                    #    print('min/max(obs_lon_data) (0):',min(obs_lon_data),max(obs_lon_data))
                    #    obs_lon_data = obs_lon_data + 180
                    #    print('min/max(obs_lon_data) (1):',min(obs_lon_data),max(obs_lon_data))
                    #    obs_lon_data = np.where(obs_lon_data > 180.0, obs_lon_data-360.0, obs_lon_data)
                    #    print('min/max(obs_lon_data) (2):',min(obs_lon_data),max(obs_lon_data))
                    ##print('obs_lat_data.shape :',obs_lat_data.shape)
                    ##print('obs_lon_data.shape :',obs_lon_data.shape)
                    #obs_latlon_data = np.concatenate((obs_lat_data,obs_lon_data),axis=1)
                    ##print('obs_latlon_data.shape :',obs_latlon_data.shape)
                    #################################################################################################
                    #################################################################################################

                    ################################################################################################
                    # This is an empty block
                    ################################################################################################
                    # temp fix
                    obs_lat_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 0:1]
                    obs_lon_data = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 1:2]
                    #if len(obs_lon_data) > 0:
                    #    print('min/max(obs_lon_data) :',min(obs_lon_data),max(obs_lon_data))
                    obs_latlon_data = np.concatenate((obs_lat_data,obs_lon_data),axis=1)
                    ################################################################################################
                    ################################################################################################

                    obs_latlon[step, j, var_starts[j]:var_starts[j]+len(obs_latlon_data)] = obs_latlon_data
                    var_starts[j] += len(all_obs_data)

            output = (torch.from_numpy(all_obs).to(device), torch.from_numpy(H_idxs).long().to(device), torch.from_numpy(H_obs).to(device),
                    torch.from_numpy(shapes).long().to(device), obs_latlon)
            return output

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_datetime <= self.end_datetime:
            obs_cumulative = self.read_file()
            self.start_datetime = self.start_datetime + timedelta(hours=self.da_window)
            self.curr_datetime = self.curr_datetime + timedelta(hours=self.da_window)
            #self.end_datetime = self.end_datetime + timedelta(hours=self.da_window)
            self.end_datetime = self.end_datetime
            return obs_cumulative
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

