import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, calendar
import torch
import inspect
sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from datetime import *
from torch.utils.data import IterableDataset, DataLoader
from itertools import product
from natsort import natsorted
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

from src.obs import *

class ERA5Data:
    def __init__(self, start_date, end_date, time_step, vars, dir = None, shards = 40, means = None,
                 stds = None, lat = None, lon = None):
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
        self.data = self.load_era5(shards = shards)
        self.varmax = np.max(self.data, axis = (0, 2, 3))
        self.varmin = np.min(self.data, axis = (0, 2, 3))

    def load_era5(self, shards = 40):
        hours_per_shard = (365 * 24) // shards
        start_date_year = datetime(self.start_date.year, 1, 1, hour = 0)
        start_shard = int(((self.start_date - start_date_year).total_seconds() // 3600) // hours_per_shard)
        start_hour = int((self.start_date -
                      (start_date_year + timedelta(hours = hours_per_shard * start_shard))).total_seconds() // 3600)
        end_date_year = datetime(self.end_date.year, 1, 1, hour = 0)
        end_shard = int(((self.end_date - end_date_year).total_seconds() // 3600) // hours_per_shard)
        years = np.arange(self.start_date.year, self.end_date.year+1, dtype = int)
        if np.any([calendar.isleap(year) for year in years]):
            raise ValueError('Date range cannot contain a leap year.')
        data = np.zeros((0, len(self.vars), self.nlat, self.nlon))
        for year in years:
            if year == self.start_date.year:
                first_shard = start_shard
            else:
                first_shard = 0
                start_hour = (first_shard * hours_per_shard) % self.time_step
            if year == self.end_date.year:
                last_shard = end_shard
            else:
                last_shard = shards
            for shard in range(first_shard, last_shard+1):
                data_f = np.load(os.path.join(self.dir, f'{year}_{shard}.npz'))
                data_in = np.zeros((data_f[self.vars[0]][start_hour::self.time_step].shape[0], len(self.vars),
                                    self.nlat, self.nlon))
                print(np.arange(data_f[self.vars[0]].shape[0])[start_hour::self.time_step] + shard * hours_per_shard)
                for i, var in enumerate(self.vars):
                    data_in[:, i] = data_f[var][start_hour::self.time_step, 0]
                data = np.concatenate((data, data_in), axis = 0)
                start_hour = self.time_step - ((shard+1) * hours_per_shard) % self.time_step
                print(start_hour)
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
            self.analysis_files = natsorted(glob.glob(os.path.join(self.dir, f'analysis_*_{self.runstr}.npy')))
            self.background_files = natsorted(glob.glob(os.path.join(self.dir, f'background_*_{self.runstr}.npy')))
            min_num_files = min(len(self.analysis_files), len(self.background_files))
            self.analysis_files = self.analysis_files[:min_num_files]
            self.background_files = self.background_files[:min_num_files]
            self.end_date = self.start_date + timedelta(hours = self.time_step * (min_num_files-1))
        else:
            cycles = (self.end_date - self.start_date).total_seconds() // 3600 // self.time_step
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
            analysis[i] = np.load(file)[0]
        for i, file in enumerate(self.background_files):
            background[i] = np.load(file)[0]

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
                 lon = None):
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
        if not file:
            self.file = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"
        obs_dataset = ObsDataset(self.file, self.start_date, self.end_date, 0, self.time_step, self.time_step,
                                 self.vars)
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
        all_x_obs = np.array((x.shape[0], x.shape[1]), dtype = object)
        for i, j in product(range(x.shape[0]), range(x.shape[1])):
            all_x_obs[i, j] = self.observe(x[i, j], i, j)
        if not return_error:
            return all_x_obs
        else:
            all_x_obs_error = np.array((x.shape[0], x.shape[1]), dtype = object)
            for i, j in product(range(x.shape[0]), range(x.shape[1])):
                all_x_obs_error[i, j] = self.obs[i][j, :self.n_obs[i][j]].detach().cpu().numpy() - \
                    all_x_obs[i,j]
            if not return_error_maxmin:
                return all_x_obs, all_x_obs_error
            else:
                err_max = np.zeros(all_x_obs.shape)
                err_min = np.zeros(all_x_obs.shape)
                for i, j in product(range(all_x_obs.shape[0]), range(all_x_obs.shape[1])):
                    err_max = np.max(all_x_obs_error[i, j])
                    err_min = np.min(all_x_obs_error[i, j])
                return all_x_obs, all_x_obs_error, np.max(err_max, axis = 0), np.min(err_min, axis = 0)

    def observe(self, x, time_idx, var_idx):
        output = observe_linear(torch.from_numpy(x).reshape(-1, 1),
                                self.H_idxs[time_idx][var_idx, :self.n_obs[time_idx][var_idx]],
                                self.H_vals[time_idx][var_idx, :self.n_obs[time_idx][var_idx]]).detach().cpu().numpy()
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
        obs_unstandardized = obs * self.stds.reshape(-1, 1) + self.means.reshape(-1, 1)
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
        self.forecasts = self.standardize()

    def load_data(self):
        if not self.end_date:
            self.forecast_files = natsorted(glob.glob(os.path.join(self.dir, f'forecast_*_{self.runstr}.npy')))
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
            forecasts[i] = np.load(file)
        return forecasts

    def unstandardize(self, means=None, stds=None):
        if means is None and self.means is None:
            raise ValueError('Means is not defined and has not been input.')
        if stds is None and self.stds is None:
            raise ValueError('Stds is not defined and has not been input.')
        if means is not None:
            means_array = np.zeros(len(self.vars))
            for i, var in enumerate(self.vars):
                means_array[i] = means[var][0]
            self.means = means_array
        if stds is not None:
            stds_array = np.zeros(len(self.vars))
            for i, var in enumerate(self.vars):
                stds_array[i] = stds[var][0]
            self.stds = stds_array
        forecasts_unstandardized = self.forecasts * self.stds.reshape(1, 1, -1, 1, 1) + \
                                   self.means.reshape(1, 1, -1, 1, 1)
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


def plot_analysis(era5, analysis, obs, show = True, save = False, figsize = (15, 7),
                  var_lim = None, err_var_lim = None, var_idxs = None, save_dir = None,
                  itr_idxs = None, zero_center_error = True, return_error = False):
    if not save and not show and not return_error:
        print('Function does not return anything, aborting...')
        return
    if save and not save_dir:
        save_dir = os.path.join(os.getcwd(), 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if not var_idxs:
        var_idxs = np.arange(analysis.shape[1], dtype = int)
    if not var_lim:
        var_max = np.maximum(np.maximum(era5.varmax, analysis.varmax), obs.varmax)
        var_min = np.minimum(np.minimum(era5.varmin, analysis.varmin), obs.varmin)
        var_lim = [(vmin, vmax) for vmin, vmax in zip(var_min, var_max)]
    era5_minus_analysis = era5.data[:analysis.analysis.shape[0]] - analysis.analysis
    if not itr_idxs:
        itr_idxs = np.arange(era5_minus_analysis.shape[0])
    era5_err_max = np.max(era5_minus_analysis, axis = (0, 2, 3))
    era5_err_min = np.min(era5_minus_analysis, axis = (0, 2, 3))
    if not err_var_lim:
        era5_obs, era5_obs_error, era5_obs_error_max, era5_obs_error_min = obs.observe_all(era5.data,
                                                                                           return_error = True,
                                                                                           return_error_maxmin = True)
        analysis_obs, analysis_obs_error, analysis_obs_error_max, analysis_obs_error_min = \
            obs.observe_all(analysis.analysis, return_error=True, return_error_maxmin=True)
        err_var_max = np.maximum(np.maximum(era5_obs_error_max, analysis_obs_error_max), era5_err_max)
        err_var_min = np.minimum(np.minimum(era5_obs_error_min, analysis_obs_error_min), era5_err_min)
        if zero_center_error:
            err_var_maxmin = np.maximum(np.abs(err_var_max), np.abs(err_var_min))
            err_var_lim = [(-vmax, vmax) for vmax in err_var_maxmin]
        else:
            err_var_lim = [(vmin, vmax) for vmin, vmax in zip(err_var_min, err_var_max)]
    else:
        era5_obs, era5_obs_error = obs.observe_all(era5.data, return_error=True)
        analysis_obs, analysis_obs_error = obs.observe_all(analysis.analysis, return_error=True)

    if save or show:
        for itr, (var_idx, var) in product(itr_idxs, [(idx, analysis.vars[idx]) for idx in var_idxs]):
            obs_latlon = obs.obs_latlon[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy()
            obs_lat_plot = obs_latlon[:, 0] + 90
            obs_lon_plot = (obs_latlon[:, 1] + 360) % 360
            fig, axs = plt.subplots(2, 3, sharex = True, sharey = True, figsize = figsize)

            pc_era5 = axs[0, 0].pcolormesh(era5.lon, era5.lat, era5.data[itr, var_idx], vmin = var_lim[var_idx][0],
                                           vmax = var_lim[var_idx][1], cmap = 'viridis')
            plt.colorbar(pc_era5, ax = axs[0,0])
            axs[0, 0].set_title('ERA5')

            pc_analysis = axs[0,1].pcolormesh(analysis.lon, analysis.lat, analysis.analysis[itr, var_idx],
                                              vmin = var_lim[var_idx][0], vmax = var_lim[var_idx][1], cmap = 'viridis')
            plt.colorbar(pc_analysis, ax = axs[0, 1])
            axs[0, 1].set_title('Analysis')

            pc_error = axs[0, 2].pcolormesh(era5.lon, era5.lat, era5_minus_analysis[itr, var_idx],
                                            vmin = err_var_lim[var_idx][0], vmax = err_var_lim[var_idx][1],
                                            cmap = 'bwr')
            plt.colorbar(pc_error, ax = axs[0, 2])
            axs[0, 2].set_title('ERA5 - Analysis')

            sp_obs = axs[1,0].scatter(obs_lon_plot, obs_lat_plot,
                                      c = obs.obs[itr][var_idx, :obs.n_obs[itr][var_idx]].detach().cpu().numpy(),
                                      vmin=var_lim[var_idx][0], vmax=var_lim[var_idx][1], cmap='viridis',
                                      edgecolor = 'k', s= 35)
            plt.colorbar(sp_obs, ax = axs[1,0])
            axs[1, 0].set_title('Observations')

            sp_era_obs = axs[1,1].scatter(obs_lon_plot, obs_lat_plot, c = era5_obs_error[itr, var_idx],
                                          vmin=err_var_lim[var_idx][0], vmax=err_var_lim[var_idx][1], cmap='bwr',
                                          edgecolor='k', s=35)
            plt.colorbar(sp_era_obs, ax=axs[1,1])
            axs[1, 1].set_title('Obs - H(ERA5)')

            sp_analysis_obs = axs[1, 2].scatter(obs_lon_plot, obs_lat_plot, c=analysis_obs_error[itr, var_idx],
                                           vmin=err_var_lim[var_idx][0], vmax=err_var_lim[var_idx][1], cmap='bwr',
                                           edgecolor='k', s=35)
            plt.colorbar(sp_analysis_obs, ax=axs[1, 2])
            axs[1, 2].set_title('Obs - H(Analysis)')

            axs[0, 0].set_ylabel('Lat')
            axs[1, 0].set_ylabel('Lat')
            axs[1, 0].set_xlabel('Lon')
            axs[1, 1].set_xlabel('Lon')
            axs[1, 2].set_xlabel('Lon')

            plot_date = analysis.start_date + timedelta(hours = itr * analysis.time_step)
            fig.suptitle(f'{var} on {plot_date.strftime("%m/%d/%Y, %H")}')
            if save:
                plt.savefig(os.path.join(save_dir, f'{var}_{itr:04}_{analysis.runstr}.png'), dpi = 400,
                            bbox_inches = 'tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

    if return_error:
        return era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error
    else:
        return






