import torch, h5py, os
import torch_harmonics as th
import numpy as np
from tqdm import tqdm
import xarray
#import metpy.calc as calc
from dv import *

"""
def divergence_vorticity(uwind, vwind, lat_lon_spacing):
    delta_np = np.load('C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\delta_y.npy')
    delta = torch.from_numpy(delta_np)
    du_dy = first_derivative(uwind, axis=0, delta = delta)
    uwind_padded = torch.nn.functional.pad(torch.nn.functional.pad(uwind.reshape(1, vwind.shape[0], vwind.shape[1]),
                                                                   (1, 1), mode = 'circular').transpose(1, 2),
                                           (1, 1), mode = 'circular').transpose(1, 2).reshape(vwind.shape[0] + 2,
                                                                                              vwind.shape[1] + 2)
    vwind_padded = torch.nn.functional.pad(torch.nn.functional.pad(vwind.reshape(1, uwind.shape[0], uwind.shape[1]),
                                                                   (1, 1), mode='circular').transpose(1, 2),
                                           (1, 1), mode='circular').transpose(1, 2).reshape(uwind.shape[0] + 2,
                                                                                            uwind.shape[1] + 2)
    du_dy = torch.gradient(uwind_padded, spacing = lat_lon_spacing[0], dim = 0)[0][1:-1, 1:-1]
    dv_dy = torch.gradient(vwind_padded, spacing = lat_lon_spacing[0], dim = 0)[0][1:-1, 1:-1]
    du_dx = spherical_ddx(uwind_padded, lat_lon_spacing[1:].reshape(-1, 1))
    dv_dx = spherical_ddx(vwind_padded, lat_lon_spacing[1:].reshape(-1, 1))
    vorticity = dv_dx - du_dy; divergence = du_dx + dv_dy
    return divergence, vorticity

def spherical_ddx(input, lon_spacing):
    return (input[1:-1, 2:] - input[1:-1, :-2])/(2 * lon_spacing)

def get_lat_lon_spacing(lat_grid, lon_delta, R_earth = 1.):
    lat_lon_spacing = np.zeros(lat_grid.size + 1)
    lat_lon_spacing[0] = (lat_grid[1] - lat_grid[0])*np.pi/180 * R_earth
    lat_lon_spacing[1:] = R_earth * np.cos(lat_grid * np.pi/180) * lon_delta * np.pi / 180
    return torch.from_numpy(lat_lon_spacing)
"""

def background_error_sh(pred, truth, sht, inv_sht):
    err_mn = sht(torch.from_numpy(truth).float() - torch.from_numpy(pred).float())
    error_hf = (truth - pred) - inv_sht(err_mn).numpy()
    return np.real(err_mn[:, :, 0].numpy()), error_hf

def uv_to_dv(truth, pred, var_dict, dv_parameter_file, means, stds, savedv = False):
    dv_f = h5py.File(dv_parameter_file, 'r')
    dx = torch.from_numpy(dv_f['delta_x'][:])
    dy = torch.from_numpy(dv_f['delta_y'][:])
    parallel_scale = torch.from_numpy(dv_f['parallel_scale'][:])
    meridional_scale = torch.from_numpy(dv_f['meridional_scale'][:])
    dx_correction = torch.from_numpy(dv_f['dx_correction'][:])
    dy_correction = torch.from_numpy(dv_f['dy_correction'][:])
    dv_f.close()

    for var in [var for var in var_dict.keys() if 'u_component_of_wind' in var]:
        if '10m' in var:
            true_divergence, true_vorticity = \
                divergence_vorticity(torch.from_numpy(truth[var_dict[var]] * stds[var][0] + means[var][0]),
                                     torch.from_numpy(truth[var_dict['10m_v_component_of_wind']] \
                                                      * stds['10m_v_component_of_wind'][0] + \
                                                      means['10m_v_component_of_wind'][0]),
                                     dx, dy, parallel_scale, meridional_scale, dx_correction, dy_correction)
            truth[var_dict[var]] = true_divergence.numpy()
            truth[var_dict['10m_v_component_of_wind']] = true_vorticity.numpy()
            pred_divergence, pred_vorticity = \
                divergence_vorticity(torch.from_numpy(pred[var_dict[var]] * stds[var][0] + means[var][0]),
                                     torch.from_numpy(pred[var_dict['10m_v_component_of_wind']] \
                                                      * stds['10m_v_component_of_wind'][0] + \
                                                      means['10m_v_component_of_wind'][0]),
                                     dx, dy, parallel_scale, meridional_scale, dx_correction, dy_correction)
            pred[var_dict[var]] = pred_divergence.numpy()
            pred[var_dict['10m_v_component_of_wind']] = pred_vorticity.numpy()
        else:
            plevel_str = var.split('wind')[1]
            true_divergence, true_vorticity = \
                divergence_vorticity(torch.from_numpy(truth[var_dict[var]] * stds[var][0] + means[var][0]),
                                     torch.from_numpy(truth[var_dict['v_component_of_wind' + plevel_str]] \
                                                      * stds['v_component_of_wind' + plevel_str][0] + \
                                                      means['v_component_of_wind' + plevel_str][0]),
                                     dx, dy, parallel_scale, meridional_scale, dx_correction, dy_correction)
            truth[var_dict[var]] = true_divergence.numpy()
            truth[var_dict['v_component_of_wind' + plevel_str]] = true_vorticity.numpy()
            if savedv and plevel_str == '_500':
                np.save('/eagle/MDClimSim/awikner/truth_dv.npy', np.stack((true_divergence.numpy(),
                                                                       true_vorticity.numpy())))
            pred_divergence, pred_vorticity = \
                divergence_vorticity(torch.from_numpy(pred[var_dict[var]] * stds[var][0] + means[var][0]),
                                     torch.from_numpy(pred[var_dict['v_component_of_wind' + plevel_str]] \
                                                      * stds['v_component_of_wind' + plevel_str][0] + \
                                                      means['v_component_of_wind' + plevel_str][0]),
                                     dx, dy, parallel_scale, meridional_scale, dx_correction, dy_correction)
            pred[var_dict[var]] = pred_divergence.numpy()
            pred[var_dict['v_component_of_wind' + plevel_str]] = pred_vorticity.numpy()
            if savedv and plevel_str == '_500':
                np.save('/eagle/MDClimSim/awikner/pred_dv.npy', np.stack((pred_divergence.numpy(),
                                                                       pred_vorticity.numpy())))
    return truth, pred

def get_err_coeffs(truth, pred, sht, inv_sht, var_dict, dv_parameter_file, means, stds, savedv = False):
    truth, pred = uv_to_dv(truth, pred, var_dict, dv_parameter_file, means, stds, savedv)
    return background_error_sh(truth, pred, sht, inv_sht)


file = '/eagle/MDClimSim/troyarcomano/ClimaX/predictions_test/forecasts.hdf5'
lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
long = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
dv_parameter_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'
save_dir = '/eagle/MDClimSim/awikner/'
#nlat = 128; nlon = 256
#lat = np.linspace(-90 + (180/nlat)/2, 90-(180/nlat)/2, nlat)
#long = np.linspace(0, 360-(360/nlon)/2, nlon)
#lat_lon_spacing = get_lat_lon_spacing(lat, long[1] - long[0])

f = h5py.File(file, 'r')
nlat = f['truth_12hr'].shape[2]; nlon = f['truth_12hr'].shape[3]
nvars = f['truth_12hr'].shape[1]; nhours = f['truth_12hr'].shape[0]

sht = th.RealSHT(nlat, nlon, grid="equiangular").to('cpu').float()
inv_sht = th.InverseRealSHT(nlat, nlon, grid="equiangular").to("cpu").float()

forecast_lead_time = 12
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
var_idxs = np.arange(len(vars))
var_dict = dict(zip(vars, var_idxs))

#nvars = len(vars); nhours = 10
#np.random.seed(10)
#test_truth = np.random.randn(nhours, nvars, nlat, nlon)
#test_pred = np.random.randn(nhours, nvars, nlat, nlon)

#truth_u_data = xarray.DataArray(test_truth[:, 1, :, :], dims = ("t", "lat", "lon"),
#                                coords = {"lat": lat, "lon": long},
#                                attrs = dict(units='m/s'))
#truth_v_data = xarray.DataArray(test_truth[:, 2, :, :], dims = ("t", "lat", "lon"),
#                                coords = {"lat": lat, "lon": long},
#                                attrs = dict(units='m/s'))
#vort = calc.vorticity(truth_u_data, truth_v_data)

output_vars = ['2m_temperature',
'10m_wind_divergence',
'10m_wind_vorticity',
'geopotential_500',
'geopotential_700',
'geopotential_850',
'geopotential_925',
'wind_divergence_250',
'wind_divergence_500',
'wind_divergence_700',
'wind_divergence_850',
'wind_divergence_925',
'wind_vorticity_250',
'wind_vorticity_500',
'wind_vorticity_700',
'wind_vorticity_850',
'wind_vorticity_925',
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
output_idxs = np.arange(len(output_vars))
output_var_dict = dict(zip(output_vars, output_idxs))

coeffs = np.zeros((nhours, nvars, nlat))
error_hf = np.zeros((nhours, nvars, nlat, nlon))
for i in tqdm(range(nhours)):
    if i == 0:
        coeffs[i], error_hf[i] = get_err_coeffs(f['truth_12hr'][i], f['pred_12hr'][i], sht, inv_sht, var_dict, dv_parameter_file, means, stds,
                                   savedv = True)
    else:
        coeffs[i], error_hf[i] = get_err_coeffs(f['truth_12hr'][i], f['pred_12hr'][i], sht, inv_sht, var_dict, dv_parameter_file, means, stds,
                                   savedv=False)
    #coeffs[i] = get_err_coeffs(test_truth[i], test_pred[i], sht, var_dict)

np.save(os.path.join(save_dir, 'background_err_sh_coeffs.npy'), coeffs)
np.save(os.path.join(save_dir, 'background_err_hf.npy'), error_hf)
print('Done')

