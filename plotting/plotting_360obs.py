import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os, sys, glob, calendar
import torch
import inspect
from datetime import *
from torch.utils.data import IterableDataset, DataLoader
from itertools import product
from natsort import natsorted
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable
import subprocess
from PIL import Image

# TODO obs cummulative!!!
#from src.obs import *
sys.path.append("/eagle/MDClimSim/mjp5595/ml4dvar/")
from src.obs_cummulative import *
import matplotlib
#matplotlib.use('TkAgg')

def rmse_diff(diff):
    return np.sqrt(np.nanmean((diff)**2))

def rmse_lat_diff(diff,lats):
    #print('diff.shape :',diff.shape)
    width = np.shape(diff)[-1]

    weights = np.cos(np.deg2rad(lats))

    weights2d = np.zeros(np.shape(diff))

    diff_squared = diff**2.0
    #weights = np.ones((10,96))

    weights2d = np.tile(weights,(width,1))
    weights2d = np.transpose(weights2d)

    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights2d)

    return np.sqrt(weighted_average)

#class ERA5Data:
#    def __init__(self, start_date, end_date, time_step, vars, dir = None, shards = 40, means = None,
#                 stds = None, lat = None, lon = None):
#        self.save_hyperparameters()
#        if lat is not None:
#            self.nlat = lat.size
#        else:
#            self.nlat = 128
#        if lon is not None:
#            self.nlon = lon.size
#        else:
#            self.nlon = 256
#        if means is not None:
#            means_array = np.zeros(len(self.vars))
#            for i, var in enumerate(vars):
#                means_array[i] = means[var][0]
#            self.means = means_array
#        if stds is not None:
#            stds_array = np.zeros(len(self.vars))
#            for i, var in enumerate(vars):
#                stds_array[i] = stds[var][0]
#            self.stds = stds_array
#        if self.dir is None:
#            self.dir = '/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/train'
#        self.data = self.load_era5(shards = shards)
#        self.varmax = np.max(self.data, axis = (0, 2, 3))
#        self.varmin = np.min(self.data, axis = (0, 2, 3))
#
#    def load_era5(self, shards = 40):
#        #hours_per_shard = (365 * 24) // shards
#        #hours_per_shard = 6
#        hours_per_shard = 12
#        start_date_year = datetime(self.start_date.year, 1, 1, hour = 0)
#        start_shard = int(((self.start_date - start_date_year).total_seconds() // 3600) // hours_per_shard)
#        start_hour = int((self.start_date -
#                      (start_date_year + timedelta(hours = hours_per_shard * start_shard))).total_seconds() // 3600)
#        end_date_year = datetime(self.end_date.year, 1, 1, hour = 0)
#        end_shard = int(((self.end_date - end_date_year).total_seconds() // 3600) // hours_per_shard)
#        years = np.arange(self.start_date.year, self.end_date.year+1, dtype = int)
#        if np.any([calendar.isleap(year) for year in years]):
#            raise ValueError('Date range cannot contain a leap year.')
#        data = np.zeros((0, len(self.vars), self.nlat, self.nlon))
#        for year in years:
#            if year == self.start_date.year:
#                first_shard = start_shard
#            else:
#                first_shard = 0
#                start_hour = (first_shard * hours_per_shard) % self.time_step
#            if year == self.end_date.year:
#                last_shard = end_shard
#            else:
#                last_shard = shards
#            #for shard in range(first_shard, last_shard+1):
#            for shard in range(first_shard, last_shard):
#                #TODO
#                #data_f = np.load(os.path.join(self.dir, f'{year}_{shard}.npz'))
#                h5file = h5py.File(os.path.join(self.dir, '{}_{:04d}.h5'.format(year,shard+1)))
#                data_f = []
#                for var in self.vars:
#                    data_f.append(h5file['input'][var])
#                data_f = np.array(data_f)
#                #data_in = np.zeros((data_f[self.vars[0]][start_hour::self.time_step].shape[0], len(self.vars),
#                #                    self.nlat, self.nlon))
#                data_in = np.zeros((1,len(self.vars),self.nlat,self.nlon))
#                #print('data_in.shape :',data_in.shape)
#                #print('data_f.shape :',data_f.shape)
#                #print(np.arange(data_f[self.vars[0]].shape[0])[start_hour::self.time_step] + shard * hours_per_shard)
#                for i, var in enumerate(self.vars):
#                #    #data_in[:, i] = data_f[var][start_hour::self.time_step, 0]
#                    data_in[0, i] = data_f[i]
#                data = np.concatenate((data, data_in), axis = 0)
#                start_hour = self.time_step - ((shard+1) * hours_per_shard) % self.time_step
#                print(start_hour)
#        return data
#
#    def standardize(self, means = None, stds = None):
#        if means is None and self.means is None:
#            raise ValueError('Means is not defined and has not been input.')
#        if stds is None and self.stds is None:
#            raise ValueError('Stds is not defined and has not been input.')
#        if means is not None:
#            means_array = np.zeros(len(self.vars))
#            for i, var in enumerate(vars):
#                means_array[i] = means[var][0]
#            self.means = means_array
#        if stds is not None:
#            stds_array = np.zeros(len(self.vars))
#            for i, var in enumerate(vars):
#                stds_array[i] = stds[var][0]
#            self.stds = stds_array
#        return (self.data - self.means.reshape(1, -1, 1, 1))/self.stds.reshape(1, -1, 1, 1)
#
#    def save_hyperparameters(self, ignore=[]):
#        """Save function arguments into class attributes.
#        """
#        frame = inspect.currentframe().f_back
#        _, _, _, local_vars = inspect.getargvalues(frame)
#        self.hparams = {k: v for k, v in local_vars.items()
#                        if k not in set(ignore + ['self']) and not k.startswith('_')}
#        for k, v in self.hparams.items():
#            setattr(self, k, v)

class ERA5Data:
    def __init__(self,
                 start_date,
                 da_window=12,
                 data_freq=6,
                 num_windows=None,
                 vars=None,
                 dir=None,
                 means=None,
                 stds=None,
                 lat=None, 
                 lon=None):
        self.save_hyperparameters()
        if lat is not None:
            self.nlat = lat.size
        else:
            self.nlat = 128
        if lon is not None:
            self.nlon = lon.size
        else:
            self.nlon = 256
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        if self.dir is None:
            self.dir = '/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/train'
        self.data = self.load_era5(start_date,da_window,data_freq,num_windows)
        self.varmax = np.max(self.data, axis = (0, 2, 3))
        self.varmin = np.min(self.data, axis = (0, 2, 3))

    def load_era5(self, start_date, da_window, data_freq, num_windows):
        start_date_year = datetime(self.start_date.year, 1, 1, hour = 0)
        dt_diff_hours = (start_date - start_date_year).total_seconds() // 3600
        era5_start_idx = dt_diff_hours // data_freq

        if calendar.isleap(self.start_date.year):
            raise ValueError('Date range cannot contain a leap year.')
        data = np.zeros((0, len(self.vars), self.nlat, self.nlon))
        for win_num in range(num_windows):
            shard = int(era5_start_idx + win_num*(da_window//data_freq))
            print('era5 file {} : {}_{:04d}.h5'.format(win_num,self.start_date.year,shard))
            h5file = h5py.File(os.path.join(self.dir, '{}_{:04d}.h5'.format(self.start_date.year,shard)))
            data_f = []
            for var in self.vars:
                data_f.append(h5file['input'][var])
            data_f = np.array(data_f)
            data_in = np.zeros((1,len(self.vars),self.nlat,self.nlon))
            for i, var in enumerate(self.vars):
                data_in[0, i] = data_f[i]
            data = np.concatenate((data, data_in), axis = 0)
        return data

    def standardize(self, means = None, stds = None):
        if means is None and self.means is None:
            raise ValueError('Means is not defined and has not been input.')
        if stds is None and self.stds is None:
            raise ValueError('Stds is not defined and has not been input.')
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        return (self.data - self.means.reshape(1, -1, 1, 1))/self.stds.reshape(1, -1, 1, 1)

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class AnalysisData:
    def __init__(self, start_date, time_step, vars, dir = None, means = None,
                 stds = None, lat = None, lon = None, runstr = None, end_date = None):
        self.save_hyperparameters()
        if lat is not None:
            self.nlat = lat.size
        else:
            self.nlat = 128
        if lon is not None:
            self.nlon = lon.size
        else:
            self.nlon = 256
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        if not self.runstr:
            self.runstr = "%dhr_%s" % (self.time_step, start_date.strftime("%m%d%Y"))
        if not self.dir:
            self.dir = '/eagle/MDClimSim/awikner/ml4dvar/data/climaX'
        self.analysis, self.background = self.load_data()
        self.analysis, self.background = self.unstandardize()
        self.varmax = np.maximum(np.max(self.analysis, axis=(0, 2, 3)), np.max(self.background, axis=(0, 2, 3)))
        self.varmin = np.minimum(np.min(self.analysis, axis=(0, 2, 3)), np.min(self.background, axis=(0, 2, 3)))

    def load_data(self):
        if not self.end_date:
            #self.analysis_files = natsorted(glob.glob(os.path.join(self.dir, f'analysis_*_{self.runstr}.npy')))
            #self.background_files = natsorted(glob.glob(os.path.join(self.dir, f'background_*_{self.runstr}.npy')))
            self.analysis_files = natsorted(glob.glob(os.path.join(self.dir, f'analysis_*_*.npy')))
            self.background_files = natsorted(glob.glob(os.path.join(self.dir, f'background_*_*.npy')))
            min_num_files = min(len(self.analysis_files), len(self.background_files))
            self.analysis_files = self.analysis_files[:min_num_files]
            self.background_files = self.background_files[:min_num_files]
            self.end_date = self.start_date + timedelta(hours = self.time_step * (min_num_files-1))
            #self.end_date = self.start_date + timedelta(hours = self.time_step * (min_num_files))
        else:
            cycles = (self.end_date - self.start_date).total_seconds() // 3600 // self.time_step
            #self.analysis_files = [os.path.join(self.dir, f'analysis_{n:04}_{self.runstr}.npy') for n in range(cycles)]
            #self.background_files = [os.path.join(self.dir, f'background_{n:04}_{self.runstr}.npy') for n in range(cycles)]
            self.analysis_files = [os.path.join(self.dir, f'analysis_{n:04}_{self.runstr}.npy') for n in range(cycles)]
            self.background_files = [os.path.join(self.dir, f'background_{n:04}_{self.runstr}.npy') for n in range(cycles)]
            try:
                assert np.all([os.path.exists(file) for file in self.analysis_files])
                assert np.all([os.path.exists(file) for file in self.background_files])
            except:
                print('Not all analysis or background files were found. Trying older formatting...')
                self.analysis_files = [os.path.join(self.dir, f'analysis_{n}_{self.runstr}.npy') for n in range(cycles)]
                self.background_files = [os.path.join(self.dir, f'background_{n}_{self.runstr}.npy') for n in range(cycles)]
        analysis = np.zeros((len(self.analysis_files), len(self.vars), self.nlat, self.nlon))
        background = np.zeros((len(self.analysis_files), len(self.vars), self.nlat, self.nlon))
        for i, file in enumerate(self.analysis_files):
            print('analysis_file {} : {}'.format(i,file))
            analysis[i] = np.load(file)[0]
        for i, file in enumerate(self.background_files):
            background[i] = np.load(file)[0]
            print('background_file {} : {}'.format(i,file))

        return analysis, background

    def unstandardize(self, means = None, stds = None):
        if means is None and self.means is None:
            raise ValueError('Means is not defined and has not been input.')
        if stds is None and self.stds is None:
            raise ValueError('Stds is not defined and has not been input.')
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        analysis_unstandardized = self.analysis * self.stds.reshape(1, -1, 1, 1) + self.means.reshape(1, -1, 1, 1)
        background_unstandardized = self.background * self.stds.reshape(1, -1, 1, 1) + self.means.reshape(1, -1, 1, 1)
        return analysis_unstandardized, background_unstandardized

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ObsData():
    def __init__(self, start_date, end_date, time_step, vars, file = None, means = None, stds = None, lat = None,
                 lon = None, device=None):
        self.save_hyperparameters()

        self.device = device
        if device is None:
            self.device = 'cpu'

        if lat is not None:
            self.nlat = lat.size
        else:
            self.nlat = 128
        if lon is not None:
            self.nlon = lon.size
        else:
            self.nlon = 256
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        if not file:
            self.file = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"
        #obs_dataset = ObsDataset(self.file, self.start_date, self.end_date, 0, self.time_step, self.time_step,
        #                         self.vars)
        obs_dataset = ObsDatasetCum(self.file, self.start_date, self.end_date, vars)

        #class ObsDataset(IterableDataset):
        #    def __init__(self, file_path, start_datetime, end_datetime, window_len, window_step, model_step, vars, obs_start_idx=0, obs_steps=1):
        #class ObsDatasetCum(IterableDataset):
        #    def __init__(self, file_path, start_datetime, end_datetime, vars, 
        #                obs_freq=3, da_window=12, obs_start_idx=0, obs_steps=1,
        #                logger=None):

        obs_dataloader = DataLoader(obs_dataset, batch_size=1, num_workers=0)
        self.obs, self.H_idxs, self.H_vals, self.n_obs, self.obs_latlon = self.read_obs(obs_dataloader)
        self.varmax, self.varmin = self.var_max_min(self.obs)

    def read_obs(self, obs_dataloader):
        obs = []; H_idxs = []; H_vals = []; n_obs = []; obs_latlon = []
        for obs_i, H_idxs_i, H_vals_i, n_obs_i, obs_latlon_i in obs_dataloader:
            obs.append(self.unstandardize(obs_i[0, 0]))
            H_idxs.append(H_idxs_i[0, 0])
            H_vals.append(H_vals_i[0, 0])
            n_obs.append(n_obs_i[0, 0])
            obs_latlon.append(obs_latlon_i[0, 0])
        return obs, H_idxs, H_vals, n_obs, obs_latlon

    def var_max_min(self, obs):
        var_max = -np.ones(len(self.vars)) * np.inf
        var_min = np.ones(len(self.vars)) * np.inf
        for obs_i in obs:
            obs_max = np.max(obs_i.detach().cpu().numpy(), axis = 1)
            obs_min = np.min(obs_i.detach().cpu().numpy(), axis = 1)
            var_max = np.maximum(var_max, obs_max)
            var_min = np.minimum(var_min, obs_min)
        return var_max, var_min

    def observe_all(self, x, return_error = False, return_error_maxmin = False):
        #print('(observe_all) x.shape :',x.shape)
        all_x_obs = np.zeros((x.shape[0], x.shape[1]), dtype = object)
        for i, j in product(range(x.shape[0]), range(x.shape[1])):
            all_x_obs[i, j] = self.observe(x[i, j], i, j)
        if not return_error:
            return all_x_obs
        else:
            all_x_obs_error = np.zeros((x.shape[0], x.shape[1]), dtype = object)
            #print('all_x_obs_error.shape :',all_x_obs_error.shape)
            for i, j in product(range(x.shape[0]), range(x.shape[1])):
                all_x_obs_error[i, j] = self.obs[i][j, :self.n_obs[i][j]].detach().cpu().numpy() - \
                    all_x_obs[i,j]
            if not return_error_maxmin:
                return all_x_obs, all_x_obs_error
            else:
                err_max = np.zeros(all_x_obs.shape)
                err_min = np.zeros(all_x_obs.shape)
                for i, j in product(range(all_x_obs.shape[0]), range(all_x_obs.shape[1])):
                    #err_max = np.max(all_x_obs_error[i, j])
                    #err_min = np.min(all_x_obs_error[i, j])
                    err = all_x_obs_error[i][j]
                    if len(err) == 0:
                        err_max = 0
                        err_min = 0
                    else:
                        err_max = np.max(err)
                        err_min = np.min(err)
                return all_x_obs, all_x_obs_error, np.max(err_max, axis = 0), np.min(err_min, axis = 0)

    def observe(self, x, time_idx, var_idx):
        #print(time_idx, var_idx)
        #print(len(self.n_obs))
        #print(self.n_obs[time_idx].shape)
        output = observe_linear(torch.from_numpy(x).reshape(-1, 1).to(self.device),
                                self.H_idxs[time_idx][var_idx, :4*self.n_obs[time_idx][var_idx]].reshape(-1, 4).T,
                                self.H_vals[time_idx][var_idx, :4*self.n_obs[time_idx][var_idx]].reshape(-1, 4)).detach().cpu().numpy()
        return output

    def unstandardize(self, obs, means=None, stds=None):
        if means is None and self.means is None:
            raise ValueError('Means is not defined and has not been input.')
        if stds is None and self.stds is None:
            raise ValueError('Stds is not defined and has not been input.')
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        obs_unstandardized = obs.detach().cpu() * self.stds.reshape(-1, 1) + self.means.reshape(-1, 1)
        return obs_unstandardized

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ForecastData():
    def __init__(self, start_date, time_step, vars, forecast_len = 20, spin_up_cycles = 9,
                 forecast_step = 1, dir = None, means = None,
                 stds = None, lat = None, lon = None, runstr = None, end_date = None):
        self.save_hyperparameters()
        if lat is not None:
            self.nlat = lat.size
        else:
            self.nlat = 128
        if lon is not None:
            self.nlon = lon.size
        else:
            self.nlon = 256
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        if not self.runstr:
            self.runstr = "%dhr_%s" % (self.time_step, start_date.strftime("%m%d%Y"))
        if not self.dir:
            self.dir = '/eagle/MDClimSim/awikner/ml4dvar/data/climaX'

        self.forecasts = self.load_data()
        print('self.forecasts done loading')
        self.forecasts = self.unstandardize(means=means,stds=stds)
        print('self.forecasts done unstandardizing')

    def load_data(self):
        if not self.end_date:
            #self.forecast_files = natsorted(glob.glob(os.path.join(self.dir, f'forecast_*_{self.runstr}.npy')))
            self.forecast_files = natsorted(glob.glob(os.path.join(self.dir, f'forecasts_*_*.h5')))
            self.end_date = self.start_date + timedelta(hours=self.time_step * (len(self.forecast_files) - 1))
        else:
            cycles = (self.end_date - self.start_date).total_seconds() // 3600 // self.time_step
            forecast_files = [os.path.join(self.dir, f'forecast_{n:04}_{self.runstr}.npy') \
                              for n in range(self.spin_up_cycles + 1, cycles, self.forecast_step)]
            try:
                assert np.all([os.path.exists(file) for file in forecast_files])
            except:
                print('Not all analysis or background files were found. Trying older formatting...')
                forecast_files = [os.path.join(self.dir, f'analysis_{n}_{self.runstr}.npy')
                                  for n in range(self.spin_up_cycles + 1, cycles, self.forecast_step)]
        forecasts = np.zeros((len(self.forecast_files), self.forecast_len, len(self.vars), self.nlat, self.nlon))
        for i, file in enumerate(self.forecast_files):
            print('forecast_file {} : {}'.format(i,file))
            #forecasts[i] = np.load(file)
            h5file = h5py.File(file)
            forecast_tmp = []
            for j in range(len(h5file.keys())):
                # TODO because we save a forecast every 6 hours
                key = (j+1)*6
                if int(key) % 12 != 0:
                    continue
                #print('key :',key)
                #print('h5file[str(key)].shape :',h5file[str(key)].shape)
                forecast_tmp.append(h5file[str(key)])
            forecast_tmp = np.array(forecast_tmp)
            #print('forecast_tmp.shape :',forecast_tmp.shape)
            forecasts[i] = forecast_tmp
        return forecasts

    def unstandardize(self, means=None, stds=None):
        if means is None and self.means is None:
            raise ValueError('Means is not defined and has not been input.')
        if stds is None and self.stds is None:
            raise ValueError('Stds is not defined and has not been input.')
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(self.vars):
                val = means[var][0]
                means_array[i] = val
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(self.vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        #forecasts_unstandardized = self.forecasts * self.stds.reshape(1, 1, -1, 1, 1) + \
        #                           self.means.reshape(1, 1, -1, 1, 1)
        forecasts_unstandardized = np.zeros_like(self.forecasts)
        for i in range(len(self.means)):
            forecasts_unstandardized[:,:,i,:,:] = self.forecasts[:,:,i,:,:] * self.stds[i]
            forecasts_unstandardized[:,:,i,:,:] = forecasts_unstandardized[:,:,i,:,:] + self.means[i]
        return forecasts_unstandardized


    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
        """
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

def plot_analysis_innovation(era5,
                             analysis,
                             obs,
                             units,
                             var_idxs = None,
                             window_idxs = None,
                             save = False,
                             show = True,
                             figsize = (15, 7),
                             var_lim = None,
                             err_var_lim = None,
                             save_dir = None,
                             zero_center_error = True,
                             return_error = False,
                             plot_obs=True):
    if not save and not show and not return_error:
        print('Function does not return anything, aborting...')
        return
    if save and not save_dir:
        save_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if var_idxs is None:
        var_idxs = np.arange(analysis.analysis.shape[1], dtype = int)
    if var_lim is None:
        var_max = np.maximum(np.maximum(era5.varmax, analysis.varmax), obs.varmax)
        var_min = np.minimum(np.minimum(era5.varmin, analysis.varmin), obs.varmin)
        var_lim = [(vmin, vmax) for vmin, vmax in zip(var_min, var_max)]
    era5_data = era5.data[:analysis.analysis.shape[0]]
    era5_minus_analysis = era5_data - analysis.analysis
    if window_idxs is None:
        window_idxs = np.arange(era5_minus_analysis.shape[0])
    era5_err_max = np.max(era5_minus_analysis, axis = (0, 2, 3))
    era5_err_min = np.min(era5_minus_analysis, axis = (0, 2, 3))
    if err_var_lim is None:
        era5_obs, era5_obs_error, era5_obs_error_max, era5_obs_error_min = obs.observe_all(era5_data,
                                                                                           return_error = True,
                                                                                           return_error_maxmin = True)
        analysis_obs, analysis_obs_error, analysis_obs_error_max, analysis_obs_error_min = \
            obs.observe_all(analysis.analysis, return_error=True, return_error_maxmin=True)

        background_obs, innovation, innovation_obs_error_max, innovation_obs_error_min = \
            obs.observe_all(analysis.background, return_error=True, return_error_maxmin=True)

        err_var_max = np.maximum(np.maximum(era5_obs_error_max, analysis_obs_error_max), era5_err_max)
        err_var_min = np.minimum(np.minimum(era5_obs_error_min, analysis_obs_error_min), era5_err_min)
        if zero_center_error:
            err_var_maxmin = np.maximum(np.abs(err_var_max), np.abs(err_var_min))
            err_var_lim = [(-vmax, vmax) for vmax in err_var_maxmin]
        else:
            err_var_lim = [(vmin, vmax) for vmin, vmax in zip(err_var_min, err_var_max)]
    else:
        era5_obs, era5_obs_error = obs.observe_all(era5_data, return_error=True)
        analysis_obs, analysis_obs_error = obs.observe_all(analysis.analysis, return_error=True)    
        background_obs, innovation = obs.observe_all(analysis.background, return_error=True)
        
    analysis_increment = analysis.analysis - analysis.background 
    if save or show:
        for var_idx, var in [(idx, analysis.vars[idx]) for idx in var_idxs]:
            gif_files = []

            increment_limit_max = 0
            inc_mse_max = 0
            mse_err_max = 0
            rmse_err_min = 0
            rmse_err_max = 0
            for itr in window_idxs:
                increment_limit_max = max(increment_limit_max,np.max(np.abs(era5_minus_analysis[itr,var_idx])))
                inc_mse_max = max(inc_mse_max,np.mean(np.square(analysis_increment[itr,var_idx])))
                mse_err_max = max(mse_err_max,np.mean(np.square(era5_minus_analysis[itr, var_idx])))
                rmse_err_min = min(rmse_err_min,rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],analysis.lat))
                rmse_err_max = max(rmse_err_max,rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],analysis.lat))

            mse_x = []
            mse_y_inc = []
            mse_y_err = []
            for itr in window_idxs:
                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                title_str = f'{var} on {plot_date.strftime("%m/%d/%Y, %H")}'
                print(title_str)
                obs_latlon = obs.obs_latlon[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
                obs_lat_plot = obs_latlon[:, 0]
                #obs_lon_plot = (obs_latlon[:, 1] + 360) % 360
                obs_lon_plot = (obs_latlon[:, 1])
                fig, axs = plt.subplots(1, 1, figsize = figsize)

                #print('axs',axs)
                pc_era5 = axs.pcolormesh(analysis.lon, analysis.lat, analysis_increment[itr, var_idx],
                                         cmap = 'seismic',
                                         norm=colors.SymLogNorm(linthresh=1,vmin=-increment_limit_max,vmax=increment_limit_max),
                                         )
                if plot_obs:
                    ra_obs = axs.scatter(obs_lon_plot, obs_lat_plot, c = innovation[itr, var_idx],
                                                cmap='seismic',
                                                norm=colors.SymLogNorm(linthresh=1,vmin=-increment_limit_max,vmax=increment_limit_max),
                                                edgecolor='k', s=35, linewidth=0.5)
                plt.colorbar(pc_era5, ax = axs, label=units[var_idx])
                axs.set_title('Analysis Increment - Cycle {}'.format(itr))
                axs.set_xticks(np.linspace(0,360,9))
                axs.set_yticks([])

                axs_mses_inc = axs.twinx()
                axs_mses_inc.tick_params(axis='y',colors='orange')
                axs_mses_inc.set_ylim(0,inc_mse_max)
                axs_mses_inc.yaxis.tick_left()
                axs_mses_inc.yaxis.set_label_position('left')
                axs_mses_inc.set_ylabel('Mean Squared Increment ({})'.format(units[var_idx]),c='orange')
                mse_x.append(itr*(359/(len(window_idxs)-1)))
                mse_y_inc.append(np.mean(np.square(analysis_increment[itr, var_idx])))
                axs_mses_inc.plot(mse_x,mse_y_inc,c='orange',linewidth=4)

                axs_mses = axs.twinx()
                axs_mses.tick_params(axis='y',colors='magenta')
                #axs_mses.set_ylim(0,mse_err_max)
                axs_mses.set_ylim(0.9*rmse_err_min,1.1*rmse_err_max)
                axs_mses.yaxis.tick_right()
                axs_mses.yaxis.set_label_position('right')
                axs_mses.set_ylabel('Lat-Weighted RMSE ({})'.format(units[var_idx]),c='magenta')
                #mse_x.append(itr*(359/(len(window_idxs)-1)))
                #mse_y_err.append(np.mean(np.square(era5_minus_analysis[itr, var_idx])))
                mse_y_err.append(rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],analysis.lat))
                axs_mses.plot(mse_x,mse_y_err,c='magenta',linewidth=4)

                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                fig.suptitle(title_str)
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                #plt.tight_layout()

                save_name = analysis.dir.split('/')[-2]
                if save:
                    #plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_analysis_increment{analysis.runstr}.png'), dpi = 200,
                    #        bbox_inches = 'tight')
                    if plot_obs:
                        plt.savefig(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_analysis_increment{analysis.runstr}.png'), dpi = 200,
                                bbox_inches = 'tight')
                        gif_files.append(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_analysis_increment{analysis.runstr}.png'))
                    else:
                        plt.savefig(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_analysis_increment{analysis.runstr}_noObs.png'), dpi = 200,
                                bbox_inches = 'tight')
                        gif_files.append(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_analysis_increment{analysis.runstr}_noObs.png'))
                if show:
                    plt.show()
                else:
                    plt.close(fig)

            if save:
                gif_imgs = []
                for gif_f in gif_files:
                    gif_imgs.append(Image.open(gif_f))

                # create extra copies of the frist and last frame
                for x in range(0, 5):
                    im = gif_imgs[0]
                    gif_imgs.insert(0,im)
                    im = gif_imgs[-1]
                    gif_imgs.append(im)

                gif_imgs[0].save(gif_files[-1].replace('.png','.gif'),
                            save_all=True, append_images=gif_imgs[1:], optimize=False, duration=500, loop=0)

    if return_error:
        return era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error

def plot_analysis_global_rmse(era5_minus_analysis,
                              era5_minus_background,
                              analysis,
                              var_names,
                              units,
                              #var_id = 3,
                              var_idxs = [0],
                              window_idxs = None, 
                              lat_weighted = False,
                              lats = None,
                              show = True,
                              save = False,
                              figsize = (15, 7),
                              save_dir = None,
                              return_error = False
                              ):
    if not save and not show and not return_error:
        print('Function does not return anything, aborting...')
        return
    if save and not save_dir:
        save_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if var_idxs is None:
        var_idxs = np.arange(len(var_names), dtype = int)

    if save or show:
        for var_idx, var in [(idx, var_names[idx]) for idx in var_idxs]:
            # TODO check which idx to use here
            #print('era5_minus_analysis.shape :',era5_minus_analysis.shape)
            length = np.shape(era5_minus_analysis)[0]
            #length = np.shape(era5_minus_analysis)[var_idx]
            rmse = np.zeros((len(window_idxs)))
            rmse_background = np.zeros(len(window_idxs))

            if lat_weighted and lats is not None:
                #print(len(lats),np.shape(era5_minus_analysis)[-2])
                assert len(lats) == np.shape(era5_minus_analysis)[-2]
            elif lat_weighted and lats is None:
                print('you need specify lats to use latitude weighted RMSE')
                return

            for itr in window_idxs:
                    if lat_weighted and lats is not None:
                        rmse[itr] = rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],lats)
                        rmse_background[itr] = rmse_lat_diff(era5_minus_background[itr,var_idx,:,:],lats)
                    else: 
                        rmse[itr] = rmse_diff(era5_minus_analysis[itr,var_idx,:,:])
                        rmse_background[itr] = rmse_diff(era5_minus_background[itr,var_idx,:,:])


            fig, axs = plt.subplots(1, 1, figsize = figsize)

            #print(rmse)
            plt.plot(window_idxs,rmse,label='Analysis')
            plt.plot(window_idxs,rmse_background,label='Background')
            if lat_weighted:
                title = 'Lat-weighted Global Analysis RMSE For {}'.format(var_names[var_idx])
            else:
                title = 'Global Analysis RMSE For {}'.format(var_names[var_idx])
            print(title)
            plt.title(title,fontsize=18)
            plt.xlabel('Cycle Number',fontsize=18)
            plt.ylabel(f'RMSE ({units[var_idx]})',fontsize=18)
            plt.legend()

            save_name = analysis.dir.split('/')[-2]
            if save:
                #plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_{analysis.runstr}_lat_rmse.png'), dpi = 400,
                #        bbox_inches = 'tight')
                plt.savefig(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_{analysis.runstr}_lat_rmse.png'), dpi = 400,
                        bbox_inches = 'tight')
            if show:
                plt.show()
            else:
                plt.close(fig)
            plt.show()

        # Plot overall RMSE as well
        ########################################################################################################################
        ########################################################################################################################
        length = np.shape(era5_minus_analysis)[0]
        #length = np.shape(era5_minus_analysis)[var_idx]
        rmse = np.zeros((length))
        rmse_background = np.zeros((length))

        if lat_weighted and lats is not None:
            #print(len(lats),np.shape(era5_minus_analysis)[-2])
            assert len(lats) == np.shape(era5_minus_analysis)[-2]
        elif lat_weighted and lats is None:
            print('you need specify lats to use latitude weighted RMSE')
            return

        for itr in window_idxs:
            for var_idx, var in enumerate(var_names):
                if lat_weighted and lats is not None:
                    rmse[itr] += rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],lats)
                    rmse_background[itr] += rmse_lat_diff(era5_minus_background[itr,var_idx,:,:],lats)
                else: 
                    rmse[itr] += rmse_diff(era5_minus_analysis[itr,var_idx,:,:])
                    rmse_background[itr] += rmse_diff(era5_minus_background[itr,var_idx,:,:])
        rmse = rmse / len(var_names)
        rmse_background = rmse_background / len(var_names)


        fig, axs = plt.subplots(1, 1, figsize = figsize)

        #print(rmse)
        plt.plot(rmse,label='Analysis')
        plt.plot(rmse_background,label='Background')
        if lat_weighted:
            title = 'Overall Lat-weighted Global Analysis RMSE'
        else:
            title = 'Overall Global Analysis RMSE'
        print(title)
        plt.title(title,fontsize=18)
        plt.xlabel('Cycle Number',fontsize=18)
        plt.ylabel(f'mean(RMSE)',fontsize=18)
        plt.legend()

        save_name = analysis.dir.split('/')[-2]
        if save:
            #plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_{analysis.runstr}_lat_rmse.png'), dpi = 400,
            #        bbox_inches = 'tight')
            plt.savefig(os.path.join(save_dir, f'{save_name}_overall_{itr:04}_{analysis.runstr}_lat_rmse.png'), dpi = 400,
                    bbox_inches = 'tight')
        if show:
            plt.show()
        else:
            plt.close(fig)
        plt.show()
        ########################################################################################################################
        ########################################################################################################################


def plot_analysis(era5,
                  analysis,
                  obs,
                  units,
                  save = False,
                  show = True,
                  figsize = (15, 7),
                  var_lim = None,
                  err_var_lim = None,
                  var_idxs = None,
                  save_dir = None,
                  window_idxs = None, 
                  zero_center_error = True, 
                  return_error = False):
    if not save and not show and not return_error:
        print('Function does not return anything, aborting...')
        return
    if save and not save_dir:
        save_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if var_idxs is None:
        var_idxs = np.arange(analysis.analysis.shape[1], dtype = int)

    era5_data = era5.data[:analysis.analysis.shape[0]]
    era5_minus_analysis = era5_data - analysis.analysis
    era5_minus_background = era5_data - analysis.background
    if window_idxs is None:
        window_idxs = np.arange(era5_minus_analysis.shape[0])
    if err_var_lim is None:
        era5_obs, era5_obs_error, era5_obs_error_max, era5_obs_error_min = obs.observe_all(era5_data,
                                                                                           return_error = True,
                                                                                           return_error_maxmin = True)
        analysis_obs, analysis_obs_error, analysis_obs_error_max, analysis_obs_error_min = \
            obs.observe_all(analysis.analysis, return_error=True, return_error_maxmin=True)
    else:
        era5_obs, era5_obs_error = obs.observe_all(era5_data, return_error=True)
        analysis_obs, analysis_obs_error = obs.observe_all(analysis.analysis, return_error=True)

    if save or show:
        for var_idx, var in [(idx, analysis.vars[idx]) for idx in var_idxs]:
            gif_files = []
            #print('era5,analysis,obs max ({}) : {}, {}, {}'.format(var,era5.varmax[var_idx], analysis.varmax[var_idx], obs.varmax[var_idx]))
            #print('era5,analysis,obs min ({}) : {}, {}, {}'.format(var,era5.varmin[var_idx], analysis.varmin[var_idx], obs.varmin[var_idx]))

            increment_limit_max = 0
            inc_mse_max = 0
            rmse_err_min = np.float('inf')
            rmse_err_max = -np.float('inf')
            for itr in window_idxs:
                increment_limit_max = max(increment_limit_max,np.max(np.abs(era5_minus_analysis[itr,var_idx])))
                inc_mse_max = max(inc_mse_max,np.mean(np.square(era5_minus_analysis[itr,var_idx])))
                rmse_err_min = min(rmse_err_min,rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],analysis.lat))
                rmse_err_max = max(rmse_err_max,rmse_lat_diff(era5_minus_analysis[itr,var_idx,:,:],analysis.lat))

            era5_vmin = np.min(era5_data[window_idxs,var_idx]) 
            era5_vmax = np.max(era5_data[window_idxs,var_idx]) 

            mse_x = []
            mse_y = []
            for itr in window_idxs:

                ###################################################################################################################################
                ###################################################################################################################################
                era5_var_data = era5_data[itr,var_idx]
                analysis_data = analysis.analysis[itr,var_idx]
                obs_data = obs.obs[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
                if len(era5_var_data)==0 or len(analysis_data)==0 or len(obs_data)==0:
                    continue
                vmin = min(np.min(era5_var_data),np.min(analysis_data),np.min(obs_data))
                vmax = max(np.max(era5_var_data),np.max(analysis_data),np.max(obs_data))

                era5_var_obs_error = era5_obs_error[itr,var_idx]
                analysis_var_obs_error = analysis_obs_error[itr,var_idx]

                all_obs_error = era5_var_obs_error + analysis_var_obs_error
                all_obs_max_error = np.max(np.abs(all_obs_error))
                counts_all, bins = np.histogram(all_obs_error, bins=51, range=(-all_obs_max_error,all_obs_max_error))

                era5_counts = np.zeros(len(counts_all))
                analysis_counts = np.zeros(len(counts_all))
                for i in range(len(era5_var_obs_error)):
                    for j,bin_lim in enumerate(bins[1:]):
                        if era5_var_obs_error[i] <= bin_lim:
                            era5_counts[j] += 1
                            break
                    for j,bin_lim in enumerate(bins[1:]):
                        if analysis_var_obs_error[i] <= bin_lim:
                            analysis_counts[j] += 1
                            break

                bar_cm = plt.cm.get_cmap('RdYlBu_r')
                bar_span = 2*all_obs_max_error
                # Gets color for each bin
                bar_CM = [bar_cm((b-min(np.min(all_obs_error),-1*np.max(all_obs_error)))/bar_span) for b in bins]
                # scale bins to lon
                scaled_bins = np.linspace(5,355,52)
                counts_max = max(np.max(era5_counts),np.max(analysis_counts))
                lat_range = max(analysis.lat)-min(analysis.lat)
                lat_min = min(analysis.lat)
                scaled_era5_counts = (era5_counts/counts_max) * lat_range * 1.0 
                scaled_analysis_counts = (analysis_counts/counts_max) * lat_range * 1.0

                #print('len(era5.lon) :',len(era5.lon))
                #print('era5.lon :',era5.lon)
                #print('len(scaled_bins) :',len(scaled_bins))
                #print('scaled_bins :',scaled_bins)
                #print('lat_range :',lat_range)
                #print('len(scaled_era5_counts) :',(scaled_era5_counts))
                #print('scaled_era5_counts :',scaled_era5_counts)
                ###################################################################################################################################
                ###################################################################################################################################


                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                title_str = f'{var} on {plot_date.strftime("%m/%d/%Y, %H")} - Cycle {itr}'
                print(title_str)
                obs_latlon = obs.obs_latlon[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
                obs_lat_plot = obs_latlon[:, 0]
                obs_lon_plot = (obs_latlon[:, 1])
                fig, axs = plt.subplots(2, 3, sharex = True, sharey = False, figsize = figsize)

                pc_era5 = axs[0, 0].pcolormesh(era5.lon, era5.lat, era5_data[itr, var_idx], vmin = era5_vmin,
                                               vmax = era5_vmax, cmap = 'viridis')
                plt.colorbar(pc_era5, ax = axs[0,0], label=units[var_idx])
                axs[0, 0].set_title('ERA5')
                axs[0, 0].set_xticks(np.linspace(0,360,9))

                pc_analysis = axs[0,1].pcolormesh(analysis.lon, analysis.lat, analysis.analysis[itr, var_idx],
                                                  vmin = vmin, vmax = vmax, cmap = 'viridis')
                plt.colorbar(pc_analysis, ax = axs[0, 1],label=units[var_idx])
                axs[0, 1].set_title('Analysis')
                axs[0, 1].set_yticklabels([])

                pc_error = axs[0, 2].pcolormesh(era5.lon, era5.lat, era5_minus_analysis[itr, var_idx],
                                                cmap = 'RdYlBu_r',
                                                norm=colors.SymLogNorm(linthresh=1,vmin=-increment_limit_max,vmax=increment_limit_max),
                                                )
                plt.colorbar(pc_error, ax = axs[0, 2], label=units[var_idx])
                axs[0, 2].set_title('ERA5 - Analysis Difference')
                axs[0, 2].set_yticks([])

                axs_mses = axs[0, 2].twinx()
                #axs_mses.set_ylim(0,inc_mse_max)
                axs_mses.set_ylim(0.9*rmse_err_min,1.1*rmse_err_max)
                axs_mses.yaxis.tick_left()
                axs_mses.yaxis.set_label_position('left')
                axs_mses.set_ylabel('Lat-Weighted RMSE ({})'.format(units[var_idx]))
                mse_x.append(itr*(359/(len(window_idxs)-1)))
                #mse_y.append(np.mean(np.square(era5_minus_analysis[itr, var_idx])))
                mse_y.append(rmse_lat_diff(era5_minus_analysis[itr, var_idx],analysis.lat))
                axs_mses.plot(mse_x,mse_y,c='k')

                sp_obs = axs[1,0].scatter(obs_lon_plot, obs_lat_plot,
                                          c = obs.obs[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy(),
                                          vmin=vmin, vmax=vmax, cmap='viridis',
                                          edgecolor = 'k', s= 35, linewidth=0.5)
                plt.colorbar(sp_obs, ax = axs[1,0], label=units[var_idx])
                axs[1, 0].set_title('Observations')

                era_err_obs = axs[1,1].bar(scaled_bins[:-1],scaled_era5_counts,color=bar_CM,width=scaled_bins[1]-scaled_bins[0],bottom=lat_min,linewidth=0.25,edgecolor='k',align='edge')
                sp_era_obs = axs[1,1].scatter(obs_lon_plot, obs_lat_plot, c = era5_obs_error[itr, var_idx],
                                              vmin=-all_obs_max_error, vmax=all_obs_max_error, cmap='PuOr_r',
                                              edgecolor='k', s=35, linewidth=0.25)
                plt.colorbar(sp_era_obs, ax=axs[1,1], label=units[var_idx])
                axs[1, 1].set_title('Observation Diff (ERA5)')
                axs[1, 1].set_yticklabels([])

                analysis_err_obs = axs[1,2].bar(scaled_bins[:-1],scaled_analysis_counts,color=bar_CM,width=scaled_bins[1]-scaled_bins[0],bottom=lat_min,linewidth=0.25,edgecolor='k',align='edge')

                sp_analysis_obs = axs[1, 2].scatter(obs_lon_plot, obs_lat_plot, c=analysis_obs_error[itr, var_idx],
                                                    vmin=-all_obs_max_error, vmax=all_obs_max_error, cmap='PuOr_r',
                                                    edgecolor='k', s=35, linewidth=0.25)
                plt.colorbar(sp_analysis_obs, ax=axs[1, 2], label=units[var_idx])
                axs[1, 2].set_title('Observation Diff (Analysis)')
                axs[1, 2].set_yticklabels([])

                axs[0, 0].set_ylabel('Lat')
                axs[1, 0].set_ylabel('Lat')
                axs[1, 0].set_xlabel('Lon')
                axs[1, 1].set_xlabel('Lon')
                axs[1, 2].set_xlabel('Lon')

                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                fig.suptitle(title_str)
    
                plt.tight_layout()

                save_name = analysis.dir.split('/')[-2]
                if save:
                    #plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_{analysis.runstr}.png'), dpi = 400,
                    #        bbox_inches = 'tight')
                    plt.savefig(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_{analysis.runstr}.png'), dpi = 400,
                            bbox_inches = 'tight')
                    gif_files.append(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_{analysis.runstr}.png'))
                if show:
                    plt.show()
                else:
                    plt.close(fig)

            if save:
                gif_imgs = []
                for gif_f in gif_files:
                    gif_imgs.append(Image.open(gif_f))

                # create extra copies of the frist and last frame
                for x in range(0, 5):
                    im = gif_imgs[0]
                    gif_imgs.insert(0,im)
                    im = gif_imgs[-1]
                    gif_imgs.append(im)

                gif_imgs[0].save(gif_files[-1].replace('.png','.gif'),
                            save_all=True, append_images=gif_imgs[1:], optimize=False, duration=500, loop=0)

    if return_error:
        return era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background
    else:
        return

def plot_background_vs_analysis(era5,
                                analysis,
                                obs,
                                units,
                                save = False,
                                show = True,
                                figsize = (23, 7),
                                var_lim = None,
                                err_var_lim = None,
                                var_idxs = None,
                                save_dir = None,
                                window_idxs = None, 
                                zero_center_error = True, 
                                return_error = False):
    if not save and not show and not return_error:
        print('Function does not return anything, aborting...')
        return
    if save and not save_dir:
        save_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if var_idxs is None:
        var_idxs = np.arange(analysis.analysis.shape[1], dtype = int)

    era5_data = era5.data[:analysis.analysis.shape[0]]
    era5_minus_analysis = era5_data - analysis.analysis
    era5_minus_background = era5_data - analysis.background
    if window_idxs is None:
        window_idxs = np.arange(era5_minus_analysis.shape[0])
    if err_var_lim is None:
        era5_obs, era5_obs_error, era5_obs_error_max, era5_obs_error_min = obs.observe_all(era5_data,
                                                                                           return_error = True,
                                                                                           return_error_maxmin = True)
        analysis_obs, analysis_obs_error, analysis_obs_error_max, analysis_obs_error_min = \
            obs.observe_all(analysis.analysis, return_error=True, return_error_maxmin=True)
        background_obs, background_obs_error, background_obs_error_max, background_obs_error_min = \
            obs.observe_all(analysis.background, return_error=True, return_error_maxmin=True)
    else:
        era5_obs, era5_obs_error = obs.observe_all(era5_data, return_error=True)
        analysis_obs, analysis_obs_error = obs.observe_all(analysis.analysis, return_error=True)
        background_obs, background_obs_error = obs.observe_all(analysis.background, return_error=True)

    if save or show:
        for var_idx, var in [(idx, analysis.vars[idx]) for idx in var_idxs]:
            gif_files = []
            #print('era5,analysis,obs max ({}) : {}, {}, {}'.format(var,era5.varmax[var_idx], analysis.varmax[var_idx], obs.varmax[var_idx]))
            #print('era5,analysis,obs min ({}) : {}, {}, {}'.format(var,era5.varmin[var_idx], analysis.varmin[var_idx], obs.varmin[var_idx]))

            fc_mse_max = 0
            obs_mse_max = 0
            bar_plot_mse_max = 0
            increment_limit_max = 0
            for itr in window_idxs:
                fc_mse_max = max(fc_mse_max,np.mean(np.square(era5_data[itr,var_idx]-analysis.background[itr, var_idx])),np.mean(np.square(era5_data[itr,var_idx]-analysis.analysis[itr, var_idx])))
                obs_mse_max = max(obs_mse_max,np.mean(np.square(era5_obs_error[itr,var_idx])),np.mean(np.square(analysis_obs_error[itr,var_idx])),np.mean(np.square(background_obs_error[itr,var_idx])))
                bar_plot_mse_max = max(bar_plot_mse_max,fc_mse_max,obs_mse_max)

                increment = analysis.analysis[itr, var_idx]-analysis.background[itr, var_idx]
                increment_limit_max = max(increment_limit_max,np.max(np.abs(increment)))

            era5_vmin = np.min(era5_data[window_idxs,var_idx]) 
            era5_vmax = np.max(era5_data[window_idxs,var_idx]) 

            for itr in window_idxs:

                ###################################################################################################################################
                ###################################################################################################################################
                era5_var_data = era5_data[itr,var_idx]
                background_data = analysis.background[itr,var_idx]
                analysis_data = analysis.analysis[itr,var_idx]
                obs_data = obs.obs[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
                if len(era5_var_data)==0 or len(analysis_data)==0 or len(obs_data)==0:
                    continue
                vmin = min(np.min(era5_var_data),np.min(background_data),np.min(analysis_data),np.min(obs_data))
                vmax = max(np.max(era5_var_data),np.max(background_data),np.max(analysis_data),np.max(obs_data))

                era5_var_obs_error = era5_obs_error[itr,var_idx]
                background_var_obs_error = background_obs_error[itr,var_idx]
                analysis_var_obs_error = analysis_obs_error[itr,var_idx]

                all_obs_error = era5_var_obs_error + background_var_obs_error + analysis_var_obs_error
                abs_obs_err_max = np.max(np.abs(all_obs_error))
                counts_all, bins = np.histogram(all_obs_error, bins=51,range=(-abs_obs_err_max,abs_obs_err_max))
                #print('bins :',bins)

                era5_counts = np.zeros(len(counts_all))
                analysis_counts = np.zeros(len(counts_all))
                background_counts = np.zeros(len(counts_all))
                for i in range(len(background_var_obs_error)):
                    for j,bin_lim in enumerate(bins[1:]):
                        if era5_var_obs_error[i] <= bin_lim:
                            era5_counts[j] += 1
                            break
                    for j,bin_lim in enumerate(bins[1:]):
                        if background_var_obs_error[i] <= bin_lim:
                            background_counts[j] += 1
                            break
                    for j,bin_lim in enumerate(bins[1:]):
                        if analysis_var_obs_error[i] <= bin_lim:
                            analysis_counts[j] += 1
                            break

                bar_cm = plt.cm.get_cmap('RdYlBu_r')
                bar_span = 2*abs_obs_err_max
                # Gets color for each bin
                bar_CM = [bar_cm((b-min(np.min(all_obs_error),-1*np.max(all_obs_error)))/bar_span) for b in bins]
                # scale bins to lon
                scaled_bins = np.linspace(5,355,52)
                counts_max = max(np.max(era5_counts),np.max(background_counts),np.max(analysis_counts))
                lat_range = max(analysis.lat)-min(analysis.lat)
                lat_min = min(analysis.lat)
                scaled_era5_counts = (era5_counts/counts_max) * lat_range * 1.0 
                scaled_background_counts = (background_counts/counts_max) * lat_range * 1.0 
                scaled_analysis_counts = (analysis_counts/counts_max) * lat_range * 1.0

                ###################################################################################################################################
                ###################################################################################################################################


                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                title_str = f'{var} on {plot_date.strftime("%m/%d/%Y, %H")} - Cycle {itr}'
                print(title_str)
                obs_latlon = obs.obs_latlon[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
                obs_lat_plot = obs_latlon[:, 0]
                obs_lon_plot = (obs_latlon[:, 1])
                fig, axs = plt.subplots(2, 4, sharex = False, sharey = False, figsize = figsize)

                pc_era5 = axs[0, 0].pcolormesh(era5.lon, era5.lat, era5_data[itr, var_idx], vmin = era5_vmin,
                                               vmax = era5_vmax, cmap = 'viridis')
                plt.colorbar(pc_era5, ax = axs[0,0], label=units[var_idx])
                axs[0, 0].set_xticks(np.linspace(0,360,9),'')
                axs[0, 0].set_title('ERA5')

                pc_background = axs[0,1].pcolormesh(analysis.lon, analysis.lat, analysis.background[itr, var_idx],
                                                  vmin = vmin, vmax = vmax, cmap = 'viridis')
                plt.colorbar(pc_background, ax = axs[0, 1],label=units[var_idx])
                axs[0, 1].set_title('Background')
                axs[0, 1].set_xticks(np.linspace(0,360,9),'')
                axs[0, 1].get_yaxis().set_ticklabels([])

                pc_background = axs[0,2].pcolormesh(analysis.lon, analysis.lat, analysis.analysis[itr, var_idx],
                                                  vmin = vmin, vmax = vmax, cmap = 'viridis')
                plt.colorbar(pc_background, ax = axs[0, 2],label=units[var_idx])
                axs[0, 2].set_xticks(np.linspace(0,360,9),'')
                axs[0, 2].get_yaxis().set_ticklabels([])
                axs[0, 2].set_title('Analysis')

                increment = analysis.analysis[itr, var_idx]-analysis.background[itr, var_idx]
                ana_inc = axs[0,3].pcolormesh(era5.lon, era5.lat, increment,
                                               cmap = 'seismic', 
                                               norm=colors.SymLogNorm(linthresh=1,vmin=-increment_limit_max,vmax=increment_limit_max))

                ra_obs = axs[0,3].scatter(obs_lon_plot, obs_lat_plot, c = background_obs_error[itr, var_idx],
                                          norm=colors.SymLogNorm(linthresh=1,vmin=-increment_limit_max,vmax=increment_limit_max),
                                          cmap='seismic', edgecolor='k', s=20, linewidth=0.25,
                                          )
                plt.colorbar(ana_inc, ax = axs[0, 3], label=units[var_idx])
                axs[0, 3].set_xticks(np.linspace(0,360,9))
                axs[0, 3].get_yaxis().set_ticklabels([])
                axs[0, 3].set_title('Analysis - Background Increment')

                
                era5_err_obs = axs[1,0].bar(scaled_bins[:-1],scaled_era5_counts,color=bar_CM,width=scaled_bins[1]-scaled_bins[0],bottom=lat_min,linewidth=0.25,edgecolor='k',align='edge')
                sp_era_obs = axs[1,0].scatter(obs_lon_plot, obs_lat_plot, c = era5_obs_error[itr, var_idx],
                                              vmin=-abs_obs_err_max, vmax=abs_obs_err_max, cmap='PuOr_r',
                                              edgecolor='k', s=35, linewidth=0.25)
                plt.colorbar(sp_era_obs, ax=axs[1,0], label=units[var_idx])
                axs[1, 0].set_xticks(np.linspace(0,360,9))
                axs[1, 0].set_title('Observation Diff (ERA5)')

                bg_err_obs = axs[1,1].bar(scaled_bins[:-1],scaled_background_counts,color=bar_CM,width=scaled_bins[1]-scaled_bins[0],bottom=lat_min,linewidth=0.25,edgecolor='k',align='edge')
                sp_era_obs = axs[1,1].scatter(obs_lon_plot, obs_lat_plot, c = background_obs_error[itr, var_idx],
                                              vmin=-abs_obs_err_max, vmax=abs_obs_err_max, cmap='PuOr_r',
                                              edgecolor='k', s=35, linewidth=0.25)
                plt.colorbar(sp_era_obs, ax=axs[1,1], label=units[var_idx])
                axs[1, 1].set_xticks(np.linspace(0,360,9))
                axs[1, 1].get_yaxis().set_ticklabels([])
                axs[1, 1].set_title('Observation Diff (Background)')

                analysis_err_obs = axs[1,2].bar(scaled_bins[:-1],scaled_analysis_counts,color=bar_CM,width=scaled_bins[1]-scaled_bins[0],bottom=lat_min,linewidth=0.25,edgecolor='k',align='edge')
                sp_analysis_obs = axs[1, 2].scatter(obs_lon_plot, obs_lat_plot, c=analysis_obs_error[itr, var_idx],
                                                    vmin=-abs_obs_err_max, vmax=abs_obs_err_max, cmap='PuOr_r',
                                                    edgecolor='k', s=35, linewidth=0.25)
                plt.colorbar(sp_analysis_obs, ax=axs[1, 2], label=units[var_idx])
                axs[1, 2].set_xticks(np.linspace(0,360,9))
                axs[1, 2].get_yaxis().set_ticklabels([])
                axs[1, 2].set_title('Observation Diff (Analysis)')

                bg_fc_mse_val = np.mean(np.square(era5_data[itr,var_idx] - analysis.background[itr, var_idx]))
                bg_obs_mse_val = np.mean(np.square(background_obs_error[itr,var_idx]))
                era5_obs_mse_val = np.mean(np.square(era5_obs_error[itr,var_idx]))
                ana_fc_mse_val = np.mean(np.square(era5_data[itr,var_idx] - analysis.analysis[itr, var_idx]))
                ana_obs_mse_val = np.mean(np.square(analysis_obs_error[itr,var_idx]))
                err_types = ('ERA5','Observation','Analysis')
                bar_plot_errs = {'Weather State Error': (0,bg_fc_mse_val,ana_fc_mse_val),
                                 'Observation Error': (era5_obs_mse_val,bg_obs_mse_val,ana_obs_mse_val),
                                 }
                bp_x = np.arange(len(err_types))  # the label locations
                width = 0.25  # the width of the bars
                multiplier = 0

                axs[1,3].set_yscale('log')
                axs[1,3].set_ylim(0, bar_plot_mse_max)
                for attribute, measurement in bar_plot_errs.items():
                    #print('attribute, measurement :',attribute,measurement)
                    offset = width * multiplier
                    rects = axs[1,3].bar(bp_x + offset, measurement, width, label=attribute)
                    #axs[1,3].bar_label(rects, padding=3, label_type='center')
                    multiplier += 1
                axs[1,3].axhline(era5_obs_mse_val,c='k',linestyle='--')
                # Add some text for labels, title and custom x-axis tick labels, etc.
                axs[1,3].set_ylabel('MSE ({})'.format(units[var_idx]))
                axs[1,3].set_title('Weather State and Observation Errors')
                axs[1,3].set_xticks(bp_x + width, err_types)
                axs[1,3].legend(loc='upper left')

                axs[0, 0].set_ylabel('Lat')
                axs[1, 0].set_ylabel('Lat')
                axs[1, 0].set_xlabel('Lon')
                axs[1, 1].set_xlabel('Lon')
                axs[1, 2].set_xlabel('Lon')

                plot_date = analysis.start_date + timedelta(hours = int(itr * analysis.time_step))
                fig.suptitle(title_str,fontsize=20)
    
                plt.tight_layout()

                save_name = analysis.dir.split('/')[-2]
                if save:
                    #plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_{analysis.runstr}.png'), dpi = 400,
                    #        bbox_inches = 'tight')
                    plt.savefig(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_{analysis.runstr}_bgVsAn.png'), dpi = 400,
                            bbox_inches = 'tight')
                    gif_files.append(os.path.join(save_dir, f'{save_name}_{var}_{itr:04}_{analysis.runstr}_bgVsAn.png'))
                if show:
                    plt.show()
                else:
                    plt.close(fig)

            if save:
                gif_imgs = []
                for gif_f in gif_files:
                    gif_imgs.append(Image.open(gif_f))

                # create extra copies of the frist and last frame
                for x in range(0, 5):
                    im = gif_imgs[0]
                    gif_imgs.insert(0,im)
                    im = gif_imgs[-1]
                    gif_imgs.append(im)

                gif_imgs[0].save(gif_files[-1].replace('.png','.gif'),
                            save_all=True, append_images=gif_imgs[1:], optimize=False, duration=500, loop=0)

    if return_error:
        return era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background
    else:
        return
