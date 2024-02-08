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
            return torch.sum((x_obs - obs)**2.0) / self.obs_err[var]

def observe_linear(x, H_idxs, H_vals):
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1),
                       axis = 1).to(device)
    return output

class ObsDataset(IterableDataset):
    def __init__(self, file_path, start_datetime, end_datetime, window_len, window_step, model_step, vars, obs_start_idx=0, obs_steps=1):
        super().__init__()
        self.save_hyperparameters()
        self.file_path = file_path
        datetime_diff = end_datetime - start_datetime
        hour_diff = datetime_diff.days*24 + datetime_diff.seconds // 3600
        self.all_obs_datetimes = [start_datetime + timedelta(hours = i) for i in \
                             range(0, hour_diff + model_step, model_step)]
        self.window_len_idxs = window_len // model_step + 1
        self.window_step_idxs = window_step // model_step
        self.num_cycles = (len(self.all_obs_datetimes) - self.window_len_idxs) // self.window_step_idxs

        self.obs_start_idx = obs_start_idx
        self.obs_steps = obs_steps

    def read_file(self):
        with h5py.File(self.file_path, 'r') as f:
            obs_datetimes = self.all_obs_datetimes[self.window_start: self.window_start + self.window_len_idxs]
            print('Obs. Datetimes')
            print(obs_datetimes)
            shapes = np.zeros((self.window_len_idxs, len(self.vars)), dtype = int)
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                if var not in f[obs_datetime.strftime("%Y/%m/%d/%H") + '/'].keys():
                    continue
                shapes[i, j] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var].shape[0]
            max_obs = np.max(shapes)
            all_obs = np.zeros((self.window_len_idxs, len(self.vars), max_obs))
            H_idxs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs), dtype = 'i4')
            H_obs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs))
            obs_latlon = np.zeros((self.window_len_idxs, len(self.vars), max_obs, 2))
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                if var not in f[obs_datetime.strftime("%Y/%m/%d/%H") + '/'].keys():
                    continue
                all_obs[i, j, :shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 2]
                H_idxs[i, j, :4*shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 0]
                H_obs[i, j, :4*shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 1]
                obs_latlon[i, j, :shapes[i,j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, :2]
            output = (torch.from_numpy(all_obs).to(device), torch.from_numpy(H_idxs).long().to(device), torch.from_numpy(H_obs).to(device),
                      torch.from_numpy(shapes).long().to(device), obs_latlon)
            return output

    def __iter__(self):
        self.window_start = -self.window_step_idxs + self.obs_start_idx
        return self

    def __next__(self):
        if self.window_start <= len(self.all_obs_datetimes) - self.window_len_idxs - 1:
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

