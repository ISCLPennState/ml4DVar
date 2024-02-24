#%%

import h5py, os, sys
from netCDF4 import Dataset
import numpy as np
import torch
from scipy.interpolate import interpn
from torch.autograd import Function
from scipy.sparse import coo_matrix, csr_matrix
from scipy.interpolate import interpn
import time

SOUNDING_TO_STORMER_pl = {'gph':'geopotential','q':'specific_humidity','temp':'temperature','uwind':'u_component_of_wind','vwind':'v_component_of_wind'}
SOUNDING_TO_STORMER_sl = {'surface_press':'mean_sea_level_pressure','surface_uwind':'10m_u_component_of_wind', 'surface_vwind':'10m_v_component_of_wind', 'surface_temp':'2m_temperature'}

def compute_H(lat_idxs, lon_idxs, lat_delta, lon_delta, lat_grid, lon_grid):
    # lat_idxs, lon_idxs -> grid_space of obs
    # lat_delta, lon_delta -> grid_space difference
    # lat_grid = np.arange(-90,90,128)
    # lon_grid = np.arange(0,360,256)

    '''
    Computes the model grid interpolated onto the observation grid and returns result and the observation operator
    :param lat_idxs:
    :param lon_idxs:
    :param lat_delta:
    :param lon_delta:
    :return:
    '''
    # TODO double check this is correct
    lat_grid_delta = np.append(lat_grid[1:] - lat_grid[:-1], 90 - lat_grid[-1] + lat_grid[0] + 90) # [1.40625]*128
    lon_grid_delta = np.append(lon_grid[1:] - lon_grid[:-1], 360 - lon_grid[-1] + lon_grid[0]) # [1.40625]*256
    H = np.zeros((lat_idxs.size, 4, 2)) # (num_obs,4,2)
    H[:, 0, 0] = np.ravel_multi_index((lat_idxs, lon_idxs), (lat_grid.size, lon_grid.size)) # finds index of lat-lon pair in flattened (128,256) vector
    H[:, 1, 0] = np.ravel_multi_index(((lat_idxs + 1) % lat_grid.size, lon_idxs), (lat_grid.size, lon_grid.size))
    H[:, 2, 0] = np.ravel_multi_index((lat_idxs, (lon_idxs + 1) % lon_grid.size), (lat_grid.size, lon_grid.size))
    H[:, 3, 0] = np.ravel_multi_index(((lat_idxs + 1) % lat_grid.size, (lon_idxs + 1) % lon_grid.size),
                                      (lat_grid.size, lon_grid.size))
    denominator = 1. / (lat_grid_delta[lat_idxs] * lon_grid_delta[lon_idxs]) # 
    H[:, 0, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta) * (lon_grid_delta[lon_idxs] - lon_delta) # H holds delta information for iterpolation
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
    # x (lat) (-90,90)
    # y (lon) (0,360)

    xi = np.searchsorted(lat, x, side='left') - 1 # lat -> x_idx
    delta_x = x - lat[xi]
    x_remove = xi == -1
    if np.any(x_remove):
        delta_x[xi == -1] = 180 + x[xi == -1] - lat[-1]
        xi[xi == -1] = len(lat) - 1
    yi = np.searchsorted(long, y, side='left') - 1 # lon -> y_idx
    delta_y = y - long[yi]
    y_remove = yi == -1
    if np.any(y_remove):
        delta_y[yi == -1] = 360 + y[yi == -1] - long[-1]
        yi[yi == -1] = len(long) - 1
    return xi, yi, delta_x, delta_y, x_remove, y_remove

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
long = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
print('Using lats :',lat)
print('Using longs :',long)

old_means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
old_stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')

# Use new means and stds
means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
print('\nUsing means :',means)
print('\nOld means :',old_means)
print('\nUsing stds :',stds)
print('\nOld stds :',old_stds)

dir = '/eagle/MDClimSim/awikner'
troy_dir = '/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/'
matt_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/obs/'

#full_obs_file = 'irga_1415_proc.hdf5'
#full_surface_obs_file = 'irga_1415_surface_proc.hdf5'
#obs_file = 'irga_1415_test1_obs.hdf5'

full_obs_file = 'irga_2014_2015_2020_all.hdf5'
msl_obs_file = 'irga_2014_2015_2020_msl_all.hdf5'
obs_file = 'igra_141520_stormer_obs_standardized_360_3.hdf5'
obs_file_raw = 'igra_141520_stormer_obs_standardized_360_3_raw.hdf5'

if os.path.exists(os.path.join(matt_dir, obs_file)):
    os.remove(os.path.join(matt_dir, obs_file))
f = h5py.File(os.path.join(dir, full_obs_file), 'r')
f_msl = h5py.File(os.path.join(dir, msl_obs_file), 'r')
f_obs = h5py.File(os.path.join(matt_dir, obs_file), 'a')
#f_obs_raw = h5py.File(os.path.join(matt_dir, obs_file_raw), 'a')
#f_surface = h5py.File(os.path.join(dir, full_surface_obs_file), 'r')

modeled_vars = ['gph', 'uwind', 'vwind', 'temp', 'q']
mean_std_names = ['geopotential', 'u_component_of_wind', 'v_component_of_wind', 'temperature', 'specific_humidity']

modeled_surface_vars = ['surface_press', 'surface_uwind', 'surface_vwind', 'surface_temp']

pred_plevels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype='f8')

for year in list(f.keys()):
    yr_grp = f_obs.create_group(year)
    for month in list(f[year].keys()):
        mth_grp = f_obs[year].create_group(month)
        for day in list(f[year + '/' + month].keys()):
            day_grp = f_obs[year + '/' + month].create_group(day)
            for hour in list(f[year + '/' + month + '/' + day].keys()):
                print(year + '/' + month + '/' + day + '/' + hour)
                hr_group = f_obs[year + '/' + month + '/' + day].create_group(hour)
                for var_idx, var in enumerate(modeled_surface_vars):
                    #print('var_idx, var (0):',var_idx, var)

                    if var == 'surface_press':
                        try:
                            obs_data = f_msl[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                        except:
                            continue
                    else:
                        try:
                            obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                        except:
                            continue
                    #obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:] # (n_obs, 4) -> (605, 4)
                    var_mean = means[f'{SOUNDING_TO_STORMER_sl[var]}'][0]
                    var_std = stds[f'{SOUNDING_TO_STORMER_sl[var]}'][0]

                    #try:
                    #    old_var_mean = old_means['%s' % (surface_vars_dict[var])][0]
                    #    old_var_std = old_stds['%s' % (surface_vars_dict[var])][0]
                    #    print('var, old_var_mean, new_var_mean :',var, old_var_mean,var_mean)
                    #    print('var, old_var_std, new_var_std :',var, old_var_std,var_std)
                    #except:
                    #    print('var, new_var_mean :',var, var_mean)
                    #    print('var, new_var_std :',var, var_std)

                    plevel_data = obs_data[:,[0,1,3]]
                    #plevel_data = plevel_data[np.lexsort((plevel_data[:, 1], plevel_data[:, 0]))]
                    #lat_obs = plevel_data[:, 0]
                    #long_obs = plevel_data[:, 1]
                    long_obs = (plevel_data[:, 1] + 360) % 360
                    plevel_data[:,1] = long_obs
                    plevel_data = plevel_data[np.lexsort((plevel_data[:, 1], plevel_data[:, 0]))]
                    lat_obs = plevel_data[:, 0]
                    long_obs = plevel_data[:,1]

                    #print('\tlong_obs (min/max):',min(long_obs),max(long_obs))
                    xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, long_obs)
                    red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                (xi != len(lat) - 1) & (yi != len(long) - 1)
                    plevel_data = plevel_data[red_idxs] # (n_obs, 3) -> (604,3)
                    delta_x = delta_x[red_idxs]
                    delta_y = delta_y[red_idxs]
                    xi_red = xi[red_idxs]
                    yi_red = yi[red_idxs]
                    plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                        '%s' % (f'{SOUNDING_TO_STORMER_sl[var]}'), data=plevel_data, dtype = 'f8'
                    )
                    plevel_dataset[:, :2] = plevel_data[:, :2]

                    #############################################################################################
                    #############################################################################################
                    # TODO should this be x100 here? Need to check final values
                    if var == 'surface_press':
                        plevel_dataset[:, 2] = (plevel_data[:, 2]*100 - var_mean)/var_std
                    elif var == 'surface_temp':
                        plevel_dataset[:, 2] = (plevel_data[:, 2] + 273.15 - var_mean) / var_std
                    else:
                        plevel_dataset[:, 2] = (plevel_data[:, 2] - var_mean) / var_std
                    plevel_dataset.attrs['mean'] = var_mean
                    plevel_dataset.attrs['std'] = var_std

                    H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, long)
                    H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                        '%s_H' % (f'{SOUNDING_TO_STORMER_sl[var]}'), H.shape, dtype = 'f8'
                    )
                    H_var_data[:] = H
                    #############################################################################################
                    #############################################################################################

                var_idx = len(modeled_surface_vars)
                for var in modeled_vars:
                    #print('var_idx, var (1):',var_idx, var)

                    try:
                        obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    except:
                        continue 
                    for plevel in pred_plevels:
                        var_mean = means[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]
                        var_std = stds[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]

                        #try:
                        #    #old_var_mean = old_means['%s' % (surface_vars_dict[var])][0]
                        #    #old_var_std = old_stds['%s' % (surface_vars_dict[var])][0]
                        #    old_var_mean = old_means[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]
                        #    old_var_std = old_stds[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]
                        #    print('old_var_mean, new_var_mean :',old_var_mean,var_mean)
                        #    print('old_var_std, new_var_std :',old_var_std,var_std)
                        #except:
                        #    print('new_var_mean :',var_mean)
                        #    print('new_var_std :',var_std)

                        plevel_data = obs_data[obs_data[:,2] == plevel]
                        plevel_data = plevel_data[:, [0,1,3]]
                        #plevel_data = plevel_data[np.lexsort((plevel_data[:,1], plevel_data[:, 0]))]
                        #lat_obs = plevel_data[:, 0]
                        long_obs = (plevel_data[:, 1] + 360) % 360
                        plevel_data[:,1] = long_obs

                        plevel_data = plevel_data[np.lexsort((plevel_data[:,1], plevel_data[:, 0]))]
                        long_obs = plevel_data[:, 1]
                        lat_obs = plevel_data[:, 0]

                        xi, yi, delta_x, delta_y, x_remove, y_remove = find_index_delta(lat_obs, long_obs)
                        red_idxs = (np.logical_not(x_remove)) & (np.logical_not(y_remove)) & \
                                    (xi != len(lat) - 1) & (yi != len(long) -1)
                        plevel_data = plevel_data[red_idxs]
                        delta_x = delta_x[red_idxs]
                        delta_y = delta_y[red_idxs]
                        xi_red = xi[red_idxs]
                        yi_red = yi[red_idxs]
                        plevel_dataset = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            (f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'), plevel_data.shape, dtype = 'f8'
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
                        H = compute_H(xi_red, yi_red, delta_x, delta_y, lat, long)
                        H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s_H' % (f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'), H.shape, dtype = 'f8'
                        )
                        H_var_data[:] = H
                        var_idx += 1

f.close()
f_obs.close()
f_msl.close()