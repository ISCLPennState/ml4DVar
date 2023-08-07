import h5py, os, sys
from netCDF4 import Dataset
import numpy as np
import torch
import torch_harmonics as th
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
    denominator = 1./(lat_grid_delta[lat_idxs] * lon_grid_delta[lon_idxs])
    H[:, 0, 1] = denominator * (lat_grid_delta[lat_idxs] - lat_delta)*(lon_grid_delta[lon_idxs] - lon_delta)
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

#n_lat = 128; n_long = 256
#n_lat =
lat = np.array([-85, 0, 85])
long = np.array([-170, -50, 50, 170])

x = np.array([-60, 10, 70])
y = np.array([-100,-10,80])



X, Y = np.meshgrid(long, lat)
print(X)
print(Y)
Z = X + Y
true_z = x + y

xi, yi, delta_x, delta_y = find_index_delta(x, y)
H_coo = compute_H(xi, yi, delta_x, delta_y, lat, long)
H = coo_matrix((H_coo[:, 1], (np.repeat(np.arange(x.size), 4), H_coo[:, 0].astype(int))),
               shape = (x.size, lat.size * long.size))
print(H.dot(Z.flatten()))
print(H.T.dot(true_z))

Z_tens = torch.from_numpy(Z.astype(float))
Z_tens.requires_grad = True

H_tens = torch.sparse_coo_tensor(np.array([H.row, H.col]), H.data, H.shape).requires_grad_(False)

print(H_tens)

out = torch.autograd.gradcheck(LinearObservation.apply, (Z_tens, H_tens), check_sparse_nnz=True)
if out:
    print('Check Grad succeeded')

n_lat = 128; n_long = 256
lat = np.linspace(-90+(n_lat/180)/2, 90-(n_lat/180)/2, n_lat)
long = np.linspace(-180+(n_long/360)/2, 180-(n_long/360)/2, n_long)

num_tests = 1000
np.random.seed(10)
Z_test = torch.rand(num_tests, n_lat, n_long)

num_obs = 1000
x_obs = np.random.rand(num_obs)*np.max(lat)*2 + np.min(lat)
y_obs = np.random.rand(num_obs)*np.max(long)*2 + np.min(long)
z_obs = x_obs + y_obs

xi, yi, delta_x, delta_y = find_index_delta(x_obs, y_obs)
H_coo = compute_H(xi, yi, delta_x, delta_y, lat, long)
H = coo_matrix((H_coo[:, 1], (np.repeat(np.arange(x_obs.size), 4), H_coo[:, 0].astype(int))),
               shape = (x_obs.size, lat.size * long.size))
H_tens = torch.sparse_coo_tensor(np.array([H.row, H.col]), H.data.astype('f4'), H.shape).to_sparse_csr().requires_grad_(False)

tic = time.perf_counter()
for i in range(num_tests):
    out = LinearObservation.apply(Z_test[i], H_tens)
toc = time.perf_counter()
print('Avg. runtime: %f sec.' % ((toc-tic)/num_tests))
print('Runtime: %f sec.' % (toc-tic))

Lat, Long = np.meshgrid(lat, long)
Z = Long + Lat
z_out = interpn((lat, long), Z.T, np.array([x_obs, y_obs]).T)
#z_out = test_interp(x_obs[:3], y_obs[:3])#[np.arange(num_obs-1, -1, -1), np.arange(num_obs)]
print(z_out - z_obs)
Z_tens_test = torch.from_numpy(Z.T.astype('f4'))
out = LinearObservation.apply(Z_tens_test, H_tens)
print(out)
"""
def residual(x, *args):
    '''

    :param x: Initial condition for the forward model. [num_vars, 128, 256]
    :param args:
    (
        background - [num_vars, 128, 256]
        background_error - scalar
        obs - [time_steps, num_vars, Max_num_obs, 3]. Each [Max_num_obs, 3] array is padded with zeros up to the
            maximum number of observations
        H_obs - [time_steps, num_vars, Max_num_obs, 4, 3]. Each [Max_num_obs, 4, 3] is padded with zeros up to the
            maximum number of observations
        num_obs - [time_steps, num_vars]
        obs_error - scalar
        obs_operator - Interpolates from model space to space of obs at each time
        nn_model - ClimaX model
        scaler - Scaler for real variables
        harmonic_scaler - Scaler for spherical harmonics
    )
    :return:
    '''
"""

