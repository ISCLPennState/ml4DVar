import numpy as np
import matplotlib.pyplot as plt
import os, sys
import torch
import inspect
sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from datetime import *
from torch.utils.data import IterableDataset, DataLoader
from natsort import natsorted
from itertools import product
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

start_date = datetime(2014, 1, 1, hour = 0)
end_date = datetime(2015, 12, 31, hour = 12)
window_len = 0
window_step = 12
model_step = 12

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
filepath = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def observe_linear(x, H_idxs, H_vals):
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1),
                       axis = 1).to(device)
    return output

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
obs_dataset = ObsDataset(filepath, start_date, end_date, window_len, window_step, model_step, vars)
print(obs_dataset.vars)
loader = DataLoader(obs_dataset, batch_size = 1, num_workers=0)

dir_climax = '/eagle/MDClimSim/awikner/climax_4dvar_troy/data/climaX'
dir_identity = '/eagle/MDClimSim/awikner/climax_4dvar_troy/data/identity'
num_cycles = 30
num_cycles_init = 30
cycles = np.arange(num_cycles)
analysis_climax = []
background_init = []
#analysis_identity = []
background_climax = []
#background_identity = []
for file in ['analysis_%d.npy' % cycle for cycle in cycles]:
    analysis_climax.append(np.load(os.path.join(dir_climax, file)))
    #analysis_identity.append(np.load(os.path.join(dir_identity, file)))
for file in ['background_%d.npy' % cycle for cycle in cycles]:
    background_climax.append(np.load(os.path.join(dir_climax, file)))
for file in ['background_init_%d.npy' % cycle for cycle in range(num_cycles_init)]:
    background_init.append(np.load(os.path.join(dir_climax, file)))
#    background_identity.append(np.load(os.path.join(dir_identity, file)))
analysis_climax = np.array(analysis_climax)
#analysis_identity = np.array(analysis_identity)
background_climax = np.array(background_climax)
background_init = np.array(background_init)
#background_identity = np.array(background_identity)

era5_1 = np.load('/eagle/MDClimSim/awikner/2014_0.npz')
era5_2 = np.load('/eagle/MDClimSim/awikner/2014_1.npz')
means = np.load('/eagle/MDClimSim/awikner/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/awikner/normalize_std.npz')

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

print(analysis_climax[0].shape)
#print(analysis_identity[0].shape)

#plt.plot(np.mean((analysis_climax - analysis_identity)**2.0, axis = (1, 2, 3, 4)))
#plt.show()



error_climax = np.zeros((cycles.size, len(vars), 128, 256))
error_climax_bkg = np.zeros((cycles.size, len(vars), 128, 256))
error_climax_bkg_init = np.zeros((num_cycles_init, len(vars), 128, 256))

obs_climax = np.zeros((cycles.size, len(vars)), dtype = object)
obs_climax_bkg = np.zeros((cycles.size, len(vars)), dtype = object)
obs_climax_bkg_init = np.zeros((cycles.size, len(vars)), dtype = object)
obs_era5 = np.zeros((cycles.size, len(vars)), dtype = object)

obs_error_climax = np.zeros((cycles.size, len(vars)), dtype = object)
obs_error_climax_bkg = np.zeros((cycles.size, len(vars)), dtype = object)
obs_error_climax_bkg_init = np.zeros((cycles.size, len(vars)), dtype = object)
obs_error_era5 = np.zeros((cycles.size, len(vars)), dtype = object)

era5 = np.zeros((era5_1['2m_temperature'].shape[0] + era5_2['2m_temperature'].shape[0], len(vars), 128, 256))

for i, var in enumerate(vars):
    era5[:, i] = np.concatenate((era5_1[var], era5_2[var]), axis = 0)

for i, var in enumerate(vars):
    era5_var = (era5[0:num_cycles*12:12, 0] - means[var][0])/stds[var][0]
    era5_var_init = (era5[0:num_cycles_init*12:12, 0] - means[var][0])/stds[var][0]
    error_climax[:, i] = era5_var - analysis_climax[:, 0, i]
    #error_identity[:, i] = era5_var - analysis_identity[:, 0, i]
    error_climax_bkg[:, i] = era5_var - background_climax[:, 0, i]
    error_climax_bkg_init[:, i] = era5_var_init - background_init[:, 0, i]
    #error_identity_bkg[:, i] = era5_var - background_identity[:, 0, i]

def rms_error(error):
    error_means = np.zeros(error.shape)
    for i, j in product(range(error_means.shape[0]), range(error_means.shape[1])):
        error_means[i, j] = np.mean(error[i,j]**2.0)
    return np.sqrt(error_means)

for cycle, (all_obs, H_idxs, H_obs, n_obs, obs_latlon) in zip(cycles, loader):
    for i, var in enumerate(vars):
        era5 = np.concatenate((era5_1[var], era5_2[var]), axis = 0)
        era5_var = (era5[cycle*12, 0] - means[var][0])/stds[var][0] 
        obs_era5 = observe_linear(torch.from_numpy(era5_var.reshape(-1,1)).to(device), H_idxs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4).T , H_obs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4))
        obs_climax = observe_linear(torch.from_numpy(analysis_climax[cycle, 0, i].reshape(-1,1)).to(device), H_idxs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4).T , H_obs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4))
        obs_climax_bkg = observe_linear(torch.from_numpy(background_climax[cycle, 0, i].reshape(-1,1)).to(device), H_idxs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4).T , H_obs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4))
        obs_climax_bkg_init = observe_linear(torch.from_numpy(background_init[cycle, 0, i].reshape(-1,1)).to(device), H_idxs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4).T , H_obs[0, 0, i, :4*n_obs[0, 0,i]].reshape(-1, 4))

        #if 'wind' in var:
        #    obs_in = (-all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()*stds[var][0] - means[var][0] - means[var][0])/stds[var][0]
        #    obs_error_era5[cycle, i] = obs_era5.detach().cpu().numpy() -obs_in
        #    obs_error_climax[cycle, i] = obs_climax.detach().cpu().numpy() - obs_in
        #    obs_error_climax_bkg[cycle, i] = obs_climax_bkg.detach().cpu().numpy() - obs_in
        #    obs_error_climax_bkg_init[cycle, i] = obs_climax_bkg_init.detach().cpu().numpy() - obs_in
        obs_error_era5[cycle, i] = obs_era5.detach().cpu().numpy() - all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()
        obs_error_climax[cycle, i] = obs_climax.detach().cpu().numpy() - all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()
        obs_error_climax_bkg[cycle, i] = obs_climax_bkg.detach().cpu().numpy() - all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()
        obs_error_climax_bkg_init[cycle, i] = obs_climax_bkg_init.detach().cpu().numpy() - all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()

        fig, axs = plt.subplots(2, 3, figsize = (15, 7), sharey = True, sharex = True)

        era5_max = np.max(era5_var*stds[var][0] + means[var][0])
        era5_min = np.min(era5_var*stds[var][0] + means[var][0])
        climax_max = np.max(analysis_climax[cycle, 0, i]*stds[var][0] + means[var][0])
        climax_min = np.min(analysis_climax[cycle, 0, i]*stds[var][0] + means[var][0])
        obs_max = np.max(all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()*stds[var][0] +means[var][0])
        obs_min = np.min(all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()*stds[var][0] +means[var][0])

        plt_max = np.max(np.array([era5_max, climax_max, obs_max]))
        plt_min = np.min(np.array([era5_min, climax_min, obs_min]))

        era5_err_max = np.max((era5_var - analysis_climax[cycle, 0, i])*stds[var][0])
        era5_err_min = np.min((era5_var - analysis_climax[cycle, 0, i])*stds[var][0])
        era5_obs_err_max = np.max(-obs_error_era5[cycle, i]*stds[var][0])
        era5_obs_err_min = np.min(-obs_error_era5[cycle, i]*stds[var][0])
        climax_obs_err_max = np.max(-obs_error_climax[cycle, i]*stds[var][0])
        climax_obs_err_min = np.min(-obs_error_climax[cycle, i]*stds[var][0])

        err_lim = np.max(np.abs(np.array([era5_err_max, era5_err_min, era5_obs_err_max, era5_obs_err_min, climax_obs_err_max, climax_obs_err_min])))


        pcm1 = axs[0,0].pcolormesh(era5_var*stds[var][0] + means[var][0], vmin = plt_min, vmax = plt_max,
                                   cmap = 'viridis')
        plt.colorbar(pcm1, ax=axs[0,0])
        pcm2 = axs[0,1].pcolormesh(analysis_climax[cycle, 0, i]*stds[var][0] + means[var][0], vmin = plt_min, vmax = plt_max, cmap = 'viridis')
        plt.colorbar(pcm2, ax=axs[0,1])
        pcm3 = axs[0,2].pcolormesh((era5_var - analysis_climax[cycle, 0, i])*stds[var][0], vmin = - err_lim, vmax = err_lim, cmap = 'bwr')
        plt.colorbar(pcm3, ax=axs[0,2])
        axs[0,0].set_title('ERA5')
        axs[0,1].set_title('Analysis')
        axs[0,2].set_title('ERA5 - Analysis')
        #print(means[var][0])
        #print(stds[var][0])
        plot_lat = (obs_latlon[0, 0, i, :n_obs[0, 0, i], 0]+90)*128/180
        plot_lon = obs_latlon[0, 0, i, :n_obs[0, 0, i], 1]
        plot_lon[plot_lon < 0] = plot_lon[plot_lon < 0] + 360
        plot_lon = plot_lon * 256/360
        scp = axs[1,0].scatter(plot_lon, plot_lat, c = all_obs[0,0,i,:n_obs[0,0,i]].detach().cpu().numpy()*stds[var][0] +means[var][0], s = 35, edgecolor='k', vmin = plt_min, vmax = plt_max, cmap = 'viridis')
        plt.colorbar(scp, ax = axs[1,0])
        #axs[1,0].set_xlabel('Lon')
        #axs[1,0].set_ylabel('Lat')
        axs[1,0].set_title('Observation')

        scp2 = axs[1,1].scatter(plot_lon, plot_lat, c = -obs_error_era5[cycle, i]*stds[var][0], s = 35, edgecolor='k', vmin = -err_lim, vmax = err_lim, cmap = 'bwr')
        plt.colorbar(scp2, ax = axs[1,1])
        axs[1,1].set_title('Obs - H(ERA5)')

        scp3 = axs[1,2].scatter(plot_lon, plot_lat, c = -obs_error_climax[cycle, i]*stds[var][0], s = 35, edgecolor='k', vmin = -err_lim, vmax = err_lim, cmap = 'bwr')
        plt.colorbar(scp3, ax = axs[1,2])
        axs[1,2].set_title('Obs - H(Analysis)')

        plt.suptitle('%s, Cycle %d' % (var, cycle))
        plt.savefig(f'/eagle/MDClimSim/awikner/climax_4dvar_troy/data/climaX/plots/cycle{cycle:04}_{var}.png', bbox_inches = 'tight')
        plt.close(fig)
        #plt.show()
    #plt.suptitle('Cycle %d' % cycle)
    #plt.show()
    #plt.clf()

    #fig = plt.figure(figsize = (7,10))
    #plt.plot(rms_error(obs_error_climax)[cycle], np.arange(len(vars)), label = 'ClimaX Analysis')
    #plt.plot(rms_error(obs_error_climax_bkg)[cycle], np.arange(len(vars)), label = 'ClimaX Background')
    #plt.plot(rms_error(obs_error_climax_bkg_init)[cycle], np.arange(len(vars)), label = 'ClimaX from Initial Background')
    #plt.plot(rms_error(obs_error_era5)[cycle], np.arange(len(vars)), label = 'ERA5')
    #plt.yticks(ticks = np.arange(len(vars)), labels = vars)
    #plt.legend()
    #plt.ylabel('Variable')
    #plt.xlabel('Observation RMSE')
    #ax = plt.gca()
    #make_axes_area_auto_adjustable(ax)
    #plt.show()

def rms_error(error):
    error_means = np.zeros(error.shape)
    for i, j in product(range(error_means.shape[0]), range(error_means.shape[1])):
        error_means[i, j] = np.mean(error[i,j]**2.0)
    return np.sqrt(np.mean(error_means, axis = 1))

plt.plot(np.sqrt(np.mean(error_climax**2.0, axis = (1, 2, 3))), label = 'ClimaX')
#plt.plot(np.mean(error_identity**2.0, axis = (1, 2, 3)), '--', label = 'Identity')
plt.plot(np.sqrt(np.mean(error_climax_bkg**2.0, axis = (1, 2, 3))),label = 'ClimaX Background')
plt.plot(np.sqrt(np.mean(error_climax_bkg_init**2.0, axis = (1, 2, 3))),label = 'ClimaX from initial background')
#plt.plot(np.mean(error_identity_bkg**2.0, axis = (1, 2, 3)), '--', label = 'Identity Background')
plt.legend()
plt.show()

plt.plot(rms_error(obs_error_climax), label = 'ClimaX Analysis')
#plt.plot(np.mean(error_identity**2.0, axis = (1, 2, 3)), '--', label = 'Identity')
plt.plot(rms_error(obs_error_climax_bkg),label = 'ClimaX Background')
plt.plot(rms_error(obs_error_climax_bkg_init),label = 'ClimaX from initial background')
plt.plot(rms_error(obs_error_era5), label = 'ERA5')
#plt.plot(np.mean(error_identity_bkg**2.0, axis = (1, 2, 3)), '--', label = 'Identity Background')
plt.legend()
plt.show()

fig, axs = plt.subplots(1, 2, figsize = (10, 5))
plt1 = axs[0].pcolormesh(analysis_climax[-1, 0, 8])
axs[0].set_title(vars[8])
fig.colorbar(plt1, ax = axs[0])
plt2 = axs[1].pcolormesh(analysis_climax[-1, 0, 13])
axs[1].set_title(vars[13])
fig.colorbar(plt2, ax = axs[1])
plt.show()


"""
end_loss = np.array(end_loss)
plt.semilogy(np.sum(end_loss, axis = 1), label = 'Total')
plt.semilogy(np.sum(end_loss[:, :27], axis = 1), label = 'Background')
plt.semilogy(np.sum(end_loss[:,27:27*2], axis = 1), label = 'Background HF')
plt.semilogy(end_loss[:, 27*2 + 8], label = 'Obs. U Wind 250')
plt.grid()
plt.legend()
plt.show()

loss_comps = []
for file in files:
    loss_comps.append(np.load(os.path.join(dir, file)))

loss_comps = np.array(loss_comps)

plt.semilogy(np.sum(loss_comps, axis = 1), label = 'Total')
plt.semilogy(np.sum(loss_comps[:, :27], axis = 1), label = 'Background')
plt.semilogy(np.sum(loss_comps[:,27:27*2], axis = 1), label = 'Background HF')
plt.semilogy(loss_comps[:, 27*2 + 8], label = 'Obs. U Wind 250')
plt.ylim(1e1, 1e7)
plt.grid()
plt.legend()
plt.show()
"""
