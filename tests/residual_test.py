import torch
import torch_harmonics as th
import os, h5py
import numpy as np
import torch
from torch.autograd import Function
from scipy.interpolate import interpn

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
    denominator = 1./(lat_grid_delta[lat_idxs] * lon_grid_delta[lon_idxs])
    H[:, 0, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta)*(lon_grid_delta[lon_idxs] - lon_delta)
    H[:, 1, 1] = denominator * (lat_delta) * (lon_grid_delta[lon_idxs] - lon_delta)
    H[:, 2, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta) * (lon_delta)
    H[:, 3, 1] = denominator * lat_delta * lon_delta
    return H.reshape(-1, 2)

def observe_linear(x, H_idxs, H_vals):
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1), axis = 1)
    return output

def find_index_delta(x, y, lat, long):
    xi = np.searchsorted(lat, x, side = 'left') - 1
    delta_x = x - lat[xi]
    if np.any(xi == -1):
        delta_x[xi == -1] = 180 + x[xi == -1] - lat[-1]
        xi[xi == -1] = len(lat) - 1
    yi = np.searchsorted(long, y, side = 'left') - 1
    delta_y = y - long[yi]
    if np.any(yi == -1):
        delta_y[yi == -1] = 360 + y[yi == -1] - long[-1]
        yi[yi == -1] = len(long) - 1
    return xi, yi, delta_x, delta_y

def fit_4dvar(x):
    optimizer = torch.optim.LBFGS([x], lr = 0.1, max_iter = 100)
    def closure():
        optimizer.zero_grad()
        loss = residual(x,
                        background,
                        background_err,
                        torch.from_numpy(all_obs),
                        torch.from_numpy(H_idxs).long(),
                        torch.from_numpy(H_obs),
                        n_obs.reshape(time_steps, nvars),
                        obs_err,
                        nn_model,
                        sht)
        loss.backward()
        return loss
    optimizer.step(closure)



def residual(x, *args):
    '''

    :param x: Initial condition for the forward model. [num_vars, 128, 256]
    :param args:
    (
        background - [num_vars, 128, 256]
        background_error - scalar
        obs - [time_steps, num_vars, Max_num_obs]. Each [Max_num_obs] array is padded with zeros up to the
            maximum number of observations
        H_idxs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        H_obs - [time_steps, num_vars, max_obs*4]. Each [Max_num_obs*4] is padded with zeros up to the
            maximum number of observations * 4
        num_obs - [time_steps, num_vars]
        obs_error - scalar
        nn_model - ClimaX model
        sht - RealSHT object for computing spherical harmonics
    )
    :return:
    '''
    background = args[0]
    background_err = args[1]
    obs = args[2]
    H_idxs = args[3]
    H_obs = args[4]
    num_obs = args[5]
    obs_err = args[6]
    nn_model = args[7]
    sht = args[8]

    num_vars = x.shape[0]
    time_steps = obs.shape[0]
    #total_obs = torch.sum(num_obs)

    # Compute background error with identity background error covariance
    x_coeffs          = sht(x)
    background_coeffs = sht(background)
    coeff_diff = x_coeffs - background_coeffs
    se_background = torch.sum(torch.abs(coeff_diff * torch.conj(coeff_diff))) / background_err

    #Compute error in observations at first time step for all variables
    se_obs = 0
    for var in range(num_vars):
        # Form sparse observation matrix from inputs. This matrix has a shape of (num_obs, nlat * nlon), and each row
        # has 4 elements corresponding to the 4 surrounding grid points on the lat - lon grid that are being
        # interpolated
        # H = torch.sparse_coo_tensor((torch.arange(num_obs[0, var]).reshape(-1, 1).expand(num_obs[0, var], 4),
        #                              H_idxs[0, var, :num_obs[0, var]*4]), H_obs[0, var, :num_obs[0, var]*4],
        #                             (num_obs[0, var], x.shape[1]*x.shape[2])).to_sparse_csr()
        x_obs = observe_linear(x.reshape(-1, 1),
                               H_idxs[0, var, :4*num_obs[0, var]].reshape(-1, 4).T,
                               H_obs[0, var, :4*num_obs[0, var]].reshape(-1, 4))
        se_obs += torch.sum((x_obs - obs[0, var, :num_obs[0, var]]) ** 2.0) / obs_err

    # Compute effects on later observations
    for step in range(1, time_steps):
        # Update model state using forward model
        x = nn_model.forward(x)
        for var in range(num_vars):
            x_obs = observe_linear(x.reshape(-1, 1),
                                   H_idxs[step, var, :4 * num_obs[step, var]].reshape(-1, 4).T,
                                   H_obs[step, var, :4 * num_obs[step, var]].reshape(-1, 4))
            se_obs += torch.sum((x_obs - obs[step, var, :num_obs[step, var]]) ** 2.0) / obs_err

    return se_obs + se_background




nlat = 4; nlon = 8
device = torch.device('cpu')
# Define SHT Object
sht = th.RealSHT(nlat, nlon, grid = "equiangular").to(device).float()
# Set lat lon grid
lat = np.linspace(-90 + (180/nlat)/2, 90-(180/nlat)/2, nlat)
long = np.linspace(0, 360-(360/nlon)/2, nlon)

# Generate pseudo-data
nvars = 2
time_steps = 2
batch_size = nvars
torch.manual_seed(10)
signal = torch.randn(batch_size, nlat, nlon).requires_grad_(True)

background_err = 0.1
background = signal.detach() + torch.randn(batch_size, nlat, nlon) * background_err

# Generate pseudo-obs and calculate H matrix for each set of obs
n_obs = torch.from_numpy(np.array([[10, 11], [12, 13]], dtype = 'i4')).flatten()
max_obs = torch.max(n_obs)
all_obs = np.zeros((time_steps, nvars, max_obs))
H_obs = np.zeros((2, 2, max_obs*4))
H_idxs = np.zeros((2, 2, max_obs*4), dtype = 'i4')
np.random.seed(10)
for i, obs in enumerate(n_obs):
    obs_lat, obs_lon = np.random.rand(obs)*lat[-1]*2 - lat[-1], np.random.rand(obs)*long[-1]
    obs_interp = interpn((lat, long), background[i % nvars].numpy(), np.array([obs_lat, obs_lon]).T)
    unraveled_idx = np.unravel_index(i, (time_steps, nvars))
    all_obs[unraveled_idx[0], unraveled_idx[1], :obs] = obs_interp
    xi, yi, delta_x, delta_y = find_index_delta(obs_lat, obs_lon, lat, long)
    H = compute_H(xi, yi, delta_x, delta_y, lat, long)
    H_obs[unraveled_idx[0], unraveled_idx[1], :4 * obs] = H[:, 1]
    H_idxs[unraveled_idx[0], unraveled_idx[1], :4 * obs] = H[:, 0].astype('i4')

obs_err = 0.1

nn_model = torch.nn.Identity()

"""
res = residual(signal,
         background,
         background_err,
         torch.from_numpy(all_obs),
         torch.from_numpy(H_idxs).long(),
         torch.from_numpy(H_obs),
         n_obs.reshape(time_steps, nvars),
         obs_err,
         nn_model,
         sht)

res.backward()

dx = signal.grad
print(dx)
"""
print(signal)
res = residual(signal,
         background,
         background_err,
         torch.from_numpy(all_obs),
         torch.from_numpy(H_idxs).long(),
         torch.from_numpy(H_obs),
         n_obs.reshape(time_steps, nvars),
         obs_err,
         nn_model,
         sht)
print(res)
fit_4dvar(signal)
print(signal)
res = residual(signal,
         background,
         background_err,
         torch.from_numpy(all_obs),
         torch.from_numpy(H_idxs).long(),
         torch.from_numpy(H_obs),
         n_obs.reshape(time_steps, nvars),
         obs_err,
         nn_model,
         sht)
print(res)
print()