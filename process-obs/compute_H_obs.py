import h5py, os, sys
from netCDF4 import Dataset
import numpy as np
import torch
from scipy.interpolate import interpn
from torch.autograd import Function
from scipy.sparse import coo_matrix, csr_matrix
from scipy.interpolate import interpn
import time


def compute_H(lat_idxs, lon_idxs, lat_delta, lon_delta, lat_grid, lon_grid):
    '''
    Computes the model grid interpolated onto the observation grid and returns result and the observation operator
    :param lat_idxs:
    :param lon_idxs:
    :param lat_delta:
    :param lon_delta:
    :return:
    '''
    lat_grid_delta = np.append(lat_grid[1:] - lat_grid[:-1], 90 - lat_grid[-1] + lat_grid[0] + 90)
    lon_grid_delta = np.append(lon_grid[1:] - lon_grid[:-1], 180 - lon_grid[-1] + lon_grid[0] + 180)
    H = np.zeros((lat_idxs.size, 4, 2))
    H[:, 0, 0] = np.ravel_multi_index((lat_idxs, lon_idxs), (lat_grid.size, lon_grid.size));
    H[:, 1, 0] = np.ravel_multi_index(((lat_idxs + 1) % lat_grid.size, lon_idxs), (lat_grid.size, lon_grid.size))
    H[:, 2, 0] = np.ravel_multi_index((lat_idxs, (lon_idxs + 1) % lon_grid.size), (lat_grid.size, lon_grid.size))
    H[:, 3, 0] = np.ravel_multi_index(((lat_idxs + 1) % lat_grid.size, (lon_idxs + 1) % lon_grid.size),
                                      (lat_grid.size, lon_grid.size))
    denominator = 1. / (lat_grid_delta[lat_idxs] * lon_grid_delta[lon_idxs])
    H[:, 0, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta) * (lon_grid_delta[lon_idxs] - lon_delta)
    H[:, 1, 1] = denominator * (lat_delta) * (lon_grid_delta[lon_idxs] - lon_delta)
    H[:, 2, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta) * (lon_delta)
    H[:, 3, 1] = denominator * lat_delta * lon_delta
    return H.reshape(-1, 2)


class LinearObservation(Function):
    @staticmethod
    def forward(ctx, x, H):
        result = torch.sparse.mm(H, x.reshape(-1, 1))
        ctx.save_for_backward(H)
        ctx.x_shape = x.shape
        return result

    @staticmethod
    def backward(ctx, grad_outputs):
        H, = ctx.saved_tensors
        return torch.sparse.mm(H.transpose(0, 1).to_sparse_csr(), grad_outputs).reshape(*ctx.x_shape), None


def find_index_delta(x, y):
    xi = np.searchsorted(lat, x, side='left') - 1
    delta_x = x - lat[xi]
    x_remove = xi == -1
    if np.any(x_remove):
        delta_x[xi == -1] = 180 + x[xi == -1] - lat[-1]
        xi[xi == -1] = len(lat) - 1
    yi = np.searchsorted(long, y, side='left') - 1
    delta_y = y - long[yi]
    y_remove = yi == -1
    if np.any(y_remove):
        delta_y[yi == -1] = 360 + y[yi == -1] - long[-1]
        yi[yi == -1] = len(long) - 1
    return xi, yi, delta_x, delta_y, x_remove, y_remove

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
long = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')

dir = '/eagle/MDClimSim/awikner'
full_obs_file = 'irga_1415_proc.hdf5'
full_surface_obs_file = 'irga_1415_surface_proc.hdf5'
obs_file = 'irga_1415_test1_obs.hdf5'
if os.path.exists(os.path.join(dir, obs_file)):
    os.remove(os.path.join(dir, obs_file))
f = h5py.File(os.path.join(dir, full_obs_file), 'r')
f_obs = h5py.File(os.path.join(dir, obs_file), 'a')
f_surface = h5py.File(os.path.join(dir, full_surface_obs_file), 'r')

modeled_vars = ['gph', 'uwind', 'vwind', 'temp', 'q']
mean_std_names = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']

surface_modeled_vars = ['temp', 'uwind', 'vwind']
surface_mean_std_names = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
vars_dict = dict(zip(modeled_vars, mean_std_names))
surface_vars_dict = dict(zip(surface_modeled_vars, surface_mean_std_names))
gph_pred_plevels = np.array([500, 700, 850, 925], dtype='f8')*100
pred_plevels     = np.array([250, 500, 700, 850, 925], dtype='f8')*100

for year in list(f_surface.keys()):
    yr_grp = f_obs.create_group(year)
    for month in list(f_surface[year].keys()):
        mth_grp = f_obs[year].create_group(month)
        for day in list(f_surface[year + '/' + month].keys()):
            day_grp = f_obs[year + '/' + month].create_group(day)
            for hour in list(f_surface[year + '/' + month + '/' + day].keys()):
                print(year + '/' + month + '/' + day + '/' + hour)
                hr_group = f_obs[year + '/' + month + '/' + day].create_group(hour)
                for var in list(f_surface[year + '/' + month + '/' + day + '/' + hour].keys()):
                    if var in surface_modeled_vars:
                        obs_data = f_surface[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                        var_mean = means['%s' % (surface_vars_dict[var])][0]
                        var_std = stds['%s' % (surface_vars_dict[var])][0]
                        plevel_data = obs_data
                        plevel_data = plevel_data[np.lexsort((plevel_data[:, 1], plevel_data[:, 0]))]
                        lat_obs = plevel_data[:, 0]
                        long_obs = plevel_data[:, 1]
                        xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, long_obs + 180)
                        red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                   (xi != len(lat) - 1) & (yi != len(long) - 1)
                        plevel_data = plevel_data[red_idxs]
                        delta_x = delta_x[red_idxs]
                        delta_y = delta_y[red_idxs]
                        xi_red = xi[red_idxs]
                        yi_red = yi[red_idxs]
                        plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s' % (surface_vars_dict[var]), (plevel_data.shape[0], 3), dtype = 'f8'
                        )
                        plevel_dataset[:, :2] = plevel_data[:, :2]
                        if var == 'gph':
                            plevel_dataset[:, 2] = (plevel_data[:, 3]*9.8 - var_mean)/var_std
                        if var == 'temp':
                            plevel_dataset[:, 2] = (plevel_data[:, 3] + 273.15 - var_mean) / var_std
                        else:
                            plevel_dataset[:, 2] = (plevel_data[:, 3] - var_mean) / var_std
                        plevel_dataset.attrs['mean'] = var_mean
                        plevel_dataset.attrs['std'] = var_std
                        H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, long)
                        H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s_H' % (surface_vars_dict[var]), H.shape, dtype = 'f8'
                        )
                        H_var_data[:] = H

for year in list(f.keys()):
    if year not in f_obs.keys():
        yr_grp = f_obs.create_group(year)
    for month in list(f[year].keys()):
        if month not in f_obs[year].keys():
            mth_grp = f_obs[year].create_group(month)
        for day in list(f[year + '/' + month].keys()):
            if day not in f_obs[year + '/' + month].keys():
                day_grp = f_obs[year + '/' + month].create_group(day)
            for hour in list(f[year + '/' + month + '/' + day].keys()):
                print(year + '/' + month + '/' + day + '/' + hour)
                if hour not in f_obs[year + '/' + month + '/' + day].keys():
                    hr_group = f_obs[year + '/' + month + '/' + day].create_group(hour)
                for var in list(f[year + '/' + month + '/' + day + '/' + hour].keys()):
                    if var in modeled_vars:
                        obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                        if var == 'gph':
                            plevels = gph_pred_plevels
                        else:
                            plevels = pred_plevels
                        for plevel in plevels:
                            var_mean = means['%s_%d' % (vars_dict[var], int(plevel)/100)][0]
                            var_std = stds['%s_%d' % (vars_dict[var], int(plevel)/100)][0]
                            plevel_data = obs_data[obs_data[:, 2] == plevel]
                            plevel_data = plevel_data[np.lexsort((plevel_data[:, 1], plevel_data[:, 0]))]
                            lat_obs = plevel_data[:, 0]
                            long_obs = plevel_data[:, 1]
                            xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, long_obs + 180)
                            red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                       (xi != len(lat) - 1) & (yi != len(long) - 1)
                            plevel_data = plevel_data[red_idxs]
                            delta_x = delta_x[red_idxs]
                            delta_y = delta_y[red_idxs]
                            xi_red = xi[red_idxs]
                            yi_red = yi[red_idxs]
                            plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                                '%s_%d' % (vars_dict[var], int(plevel)/100), (plevel_data.shape[0], 3), dtype = 'f8'
                            )
                            plevel_dataset[:, :2] = plevel_data[:, :2]
                            if var == 'gph':
                                plevel_dataset[:, 2] = (plevel_data[:, 3]*9.8 - var_mean)/var_std
                            else:
                                plevel_dataset[:, 2] = (plevel_data[:, 3] - var_mean) / var_std
                            plevel_dataset.attrs['mean'] = var_mean
                            plevel_dataset.attrs['std'] = var_std
                            H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, long)
                            H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                                '%s_%d_H' % (vars_dict[var], int(plevel)/100), H.shape, dtype = 'f8'
                            )
                            H_var_data[:] = H

f.close()
f_obs.close()
f_surface.close()
