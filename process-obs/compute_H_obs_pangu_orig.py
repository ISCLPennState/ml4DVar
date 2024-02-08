import h5py, os, sys
from netCDF4 import Dataset
import numpy as np
import torch
from scipy.interpolate import interpn
from torch.autograd import Function
from scipy.sparse import coo_matrix, csr_matrix
from scipy.interpolate import interpn
import time
import copy


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
    lon_grid_delta = np.append(lon_grid[1:] - lon_grid[:-1], 360 - lon_grid[-1] + lon_grid[0])
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
    yi = np.searchsorted(lon, y, side='left') - 1
    delta_y = y - lon[yi]
    y_remove = yi == -1
    if np.any(y_remove):
        delta_y[yi == -1] = 360 + y[yi == -1] - lon[-1]
        yi[yi == -1] = len(lon) - 1
    return xi, yi, delta_x, delta_y, x_remove, y_remove

#lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
#lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
#means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
#stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
lat = np.arange(-90,90.25,0.25)
lon = np.arange(0,360,0.25)
vars = np.load('/eagle/MDClimSim/awikner/ml4dvar/var_list_pangu.npy')
means_list = np.load('/eagle/MDClimSim/awikner/ml4dvar/pangu_means.npy')
stds_list = np.load('/eagle/MDClimSim/awikner/ml4dvar/pangu_stds.npy')
means = dict(zip(vars, [np.array([mean]) for mean in means_list]))
stds = dict(zip(vars, [np.array([std]) for std in stds_list]))
print(means)
print(stds)

dir = '/eagle/MDClimSim/awikner'
full_obs_file = 'irga_2014_2015_2020_all.hdf5'
msl_obs_file = 'irga_2014_2015_2020_msl_all.hdf5'
#full_surface_obs_file = 'irga_1415_surface_proc.hdf5'
obs_file = 'igra_141520_pangu_obs_standardized.hdf5'
if os.path.exists(os.path.join(dir, obs_file)):
    os.remove(os.path.join(dir, obs_file))
f = h5py.File(os.path.join(dir, full_obs_file), 'r')
f_msl = h5py.File(os.path.join(dir, msl_obs_file), 'r')
f_obs = h5py.File(os.path.join(dir, obs_file), 'a')

modeled_vars = ['gph','q','temp','uwind','vwind']
modeled_surface_vars = ['surface_press', 'surface_uwind', 'surface_vwind', 'surface_temp']

pred_plevels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype='f8')

for year in list(f.keys()):
    yr_grp = f_obs.create_group(year)
    for month in list(f[year].keys()):
        mth_grp = f_obs[year].create_group(month)
        for day in list(f[year + '/' + month].keys()):
            day_grp = f_obs[year + '/' + month].create_group(day)
            for hour in ['00','12']:
                print(year + '/' + month + '/' + day + '/' + hour)
                hr_group = f_obs[year + '/' + month + '/' + day].create_group(hour)
                for var_idx, var in enumerate(modeled_surface_vars):
                    if var == 'surface_press':
                        obs_data = f_msl[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    else:
                        obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    var_mean = means[vars[var_idx]][0]
                    var_std = stds[vars[var_idx]][0]
                    plevel_data = obs_data[:,[0,1,3]]
                    plevel_data = plevel_data[np.lexsort((plevel_data[:,1], plevel_data[:, 0]))]
                    lat_obs = plevel_data[:, 0]
                    lon_obs = (plevel_data[:, 1] + 360) % 360
                    plevel_data[:,1] = lon_obs
                    xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, lon_obs)
                    red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                   (xi != len(lat) - 1)
                    plevel_data = plevel_data[red_idxs]
                    delta_x = delta_x[red_idxs]
                    delta_y = delta_y[red_idxs]
                    xi_red = xi[red_idxs]
                    yi_red = yi[red_idxs]
                    plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            vars[var_idx], plevel_data.shape, dtype = 'f8'
                        )
                    plevel_dataset[:,:2] = plevel_data[:,:2]
                    if var == 'surface_press':
                        plevel_dataset[:, 2] = (plevel_data[:,2]*100 - var_mean)/var_std
                    elif var == 'surface_temp':
                        plevel_dataset[:, 2] = (plevel_data[:,2] + 273.15 - var_mean)/var_std
                    else:
                        plevel_dataset[:, 2] = (plevel_data[:,2] - var_mean)/var_std
                    plevel_dataset.attrs['mean'] = var_mean
                    plevel_dataset.attrs['std'] = var_std
                    H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, lon)
                    H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s_H' % (vars[var_idx]), H.shape, dtype = 'f8'
                        )
                    H_var_data[:] = H
                
                var_idx = len(modeled_surface_vars)
                for var in modeled_vars:
                    obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    for plevel in pred_plevels:
                        var_mean = means[vars[var_idx]][0]
                        var_std = stds[vars[var_idx]][0]
                        plevel_data = obs_data[obs_data[:,2] == plevel]
                        plevel_data = plevel_data[:, [0,1,3]]
                        plevel_data = plevel_data[np.lexsort((plevel_data[:,1], plevel_data[:, 0]))]
                        lat_obs = plevel_data[:, 0]
                        lon_obs = (plevel_data[:, 1] + 360) % 360
                        plevel_data[:,1] = lon_obs
                        xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, lon_obs)
                        red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                   (xi != len(lat) - 1)
                        plevel_data = plevel_data[red_idxs]
                        delta_x = delta_x[red_idxs]
                        delta_y = delta_y[red_idxs]
                        xi_red = xi[red_idxs]
                        yi_red = yi[red_idxs]
                        plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            vars[var_idx], plevel_data.shape, dtype = 'f8'
                            )
                        plevel_dataset[:,:2] = plevel_data[:,:2]
                        if var == 'gph':
                            plevel_dataset[:, 2] = (plevel_data[:, 2]*9.8 - var_mean)/var_std
                        elif var == 'temp':
                            plevel_dataset[:, 2] = (plevel_data[:, 2] + 273.15 - var_mean)/var_std
                        else:
                            plevel_dataset[:, 2] = (plevel_data[:, 2] - var_mean)/var_std
                        plevel_dataset.attrs['mean'] = var_mean
                        plevel_dataset.attrs['std'] = var_std
                        H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, lon)
                        H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s_H' % (vars[var_idx]), H.shape, dtype = 'f8'
                        )
                        H_var_data[:] = H
                        var_idx += 1

f.close()
f_msl.close()
f_obs.close()
