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

SOUNDING_TO_STORMER_pl = {'gph':'geopotential','q':'specific_humidity','temp':'temperature','uwind':'u_component_of_wind','vwind':'v_component_of_wind'}
SOUNDING_TO_STORMER_sl = {'surface_press':'mean_sea_level_pressure','surface_uwind':'10m_u_component_of_wind', 'surface_vwind':'10m_v_component_of_wind', 'surface_temp':'2m_temperature'}

vars = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "geopotential_50",
    "geopotential_100",
    "geopotential_150",
    "geopotential_200",
    "geopotential_250",
    "geopotential_300",
    "geopotential_400",
    "geopotential_500",
    "geopotential_600",
    "geopotential_700",
    "geopotential_850",
    "geopotential_925",
    "geopotential_1000",
    "u_component_of_wind_50",
    "u_component_of_wind_100",
    "u_component_of_wind_150",
    "u_component_of_wind_200",
    "u_component_of_wind_250",
    "u_component_of_wind_300",
    "u_component_of_wind_400",
    "u_component_of_wind_500",
    "u_component_of_wind_600",
    "u_component_of_wind_700",
    "u_component_of_wind_850",
    "u_component_of_wind_925",
    "u_component_of_wind_1000",
    "v_component_of_wind_50",
    "v_component_of_wind_100",
    "v_component_of_wind_150",
    "v_component_of_wind_200",
    "v_component_of_wind_250",
    "v_component_of_wind_300",
    "v_component_of_wind_400",
    "v_component_of_wind_500",
    "v_component_of_wind_600",
    "v_component_of_wind_700",
    "v_component_of_wind_850",
    "v_component_of_wind_925",
    "v_component_of_wind_1000",
    #"vertical_velocity_50",
    #"vertical_velocity_100",
    #"vertical_velocity_150",
    #"vertical_velocity_200",
    #"vertical_velocity_250",
    #"vertical_velocity_300",
    #"vertical_velocity_400",
    #"vertical_velocity_500",
    #"vertical_velocity_600",
    #"vertical_velocity_700",
    #"vertical_velocity_850",
    #"vertical_velocity_925",
    #"vertical_velocity_1000", # unmeasurable
    "temperature_50",
    "temperature_100",
    "temperature_150",
    "temperature_200",
    "temperature_250",
    "temperature_300",
    "temperature_400",
    "temperature_500",
    "temperature_600",
    "temperature_700",
    "temperature_850",
    "temperature_925",
    "temperature_1000",
    "specific_humidity_50",
    "specific_humidity_100",
    "specific_humidity_150",
    "specific_humidity_200",
    "specific_humidity_250",
    "specific_humidity_300",
    "specific_humidity_400",
    "specific_humidity_500",
    "specific_humidity_600",
    "specific_humidity_700",
    "specific_humidity_850",
    "specific_humidity_925",
    "specific_humidity_1000",
    ]

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
lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')

means_list = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
stds_list = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
means = means_list#dict(zip(vars, [np.array([mean]) for mean in means_list]))
stds = stds_list #dict(zip(vars, [np.array([std]) for std in stds_list]))
print(means)
print(stds)

dir = '/eagle/MDClimSim/awikner'
troy_dir = '/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/'
full_obs_file = 'irga_2014_2015_2020_all.hdf5'
msl_obs_file = 'irga_2014_2015_2020_msl_all.hdf5'
#full_surface_obs_file = 'irga_1415_surface_proc.hdf5'
obs_file = 'igra_141520_stormer_obs_standardized.hdf5'
if os.path.exists(os.path.join(troy_dir, obs_file)):
    os.remove(os.path.join(troy_dir, obs_file))
f = h5py.File(os.path.join(dir, full_obs_file), 'r')
f_msl = h5py.File(os.path.join(dir, msl_obs_file), 'r')
f_obs = h5py.File(os.path.join(troy_dir, obs_file), 'a')

modeled_vars = ['gph','q','temp','uwind','vwind']
modeled_surface_vars = ['surface_press', 'surface_uwind', 'surface_vwind', 'surface_temp']

pred_plevels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50], dtype='f8')

for year in list(f.keys()):
    yr_grp = f_obs.create_group(year)
    for month in list(f[year].keys()):
        mth_grp = f_obs[year].create_group(month)
        for day in list(f[year + '/' + month].keys()):
            day_grp = f_obs[year + '/' + month].create_group(day)
            for hour in list(f[year + '/' + month + '/' + day]):
                print(year + '/' + month + '/' + day + '/' + hour)
                hr_group = f_obs[year + '/' + month + '/' + day].create_group(hour)
                for var_idx, var in enumerate(modeled_surface_vars):
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
                    var_mean = means[f'{SOUNDING_TO_STORMER_sl[var]}'][0]
                    var_std = stds[f'{SOUNDING_TO_STORMER_sl[var]}'][0]
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
                    print('H.shape :',H.shape)
                    H_var_data = f_obs[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                            '%s_H' % (vars[var_idx]), H.shape, dtype = 'f8'
                        )
                    H_var_data[:] = H
                
                var_idx = len(modeled_surface_vars)
                for var in modeled_vars:
                    try:
                        obs_data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    except:
                        continue 
                    for plevel in pred_plevels:
                        var_mean = means[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]
                        var_std = stds[f'{SOUNDING_TO_STORMER_pl[var]}_{int(plevel)}'][0]
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
