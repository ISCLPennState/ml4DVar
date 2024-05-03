import torch
import numpy as np
import h5py
import torch_harmonics as th

def make_take(ndims, slice_dim):
    """Generate a take function to index in a particular dimension."""
    def take(indexer):
        return tuple(indexer if slice_dim % ndims == i else slice(None)  # noqa: S001
                     for i in range(ndims))
    return take

def first_derivative(f, axis=None, x=None, delta=None):
    """Calculate the first derivative of a grid of values.

    Works for both regularly-spaced data and grids with varying spacing.

    Either `x` or `delta` must be specified, or `f` must be given as an `xarray.DataArray` with
    attached coordinate and projection information. If `f` is an `xarray.DataArray`, and `x` or
    `delta` are given, `f` will be converted to a `pint.Quantity` and the derivative returned
    as a `pint.Quantity`, otherwise, if neither `x` nor `delta` are given, the attached
    coordinate information belonging to `axis` will be used and the derivative will be returned
    as an `xarray.DataArray`.

    This uses 3 points to calculate the derivative, using forward or backward at the edges of
    the grid as appropriate, and centered elsewhere. The irregular spacing is handled
    explicitly, using the formulation as specified by [Bowen2005]_.

    Parameters
    ----------
    f : array-like
        Array of values of which to calculate the derivative
    axis : int or str, optional
        The array axis along which to take the derivative. If `f` is ndarray-like, must be an
        integer. If `f` is a `DataArray`, can be a string (referring to either the coordinate
        dimension name or the axis type) or integer (referring to axis number), unless using
        implicit conversion to `pint.Quantity`, in which case it must be an integer. Defaults
        to 0. For reference, the current standard axis types are 'time', 'vertical', 'y', and
        'x'.
    x : array-like, optional
        The coordinate values corresponding to the grid points in `f`
    delta : array-like, optional
        Spacing between the grid points in `f`. Should be one item less than the size
        of `f` along `axis`.

    Returns
    -------
    array-like
        The first derivative calculated along the selected axis


    .. versionchanged:: 1.0
       Changed signature from ``(f, **kwargs)``

    See Also
    --------
    second_derivative

    """
    #n, axis, delta = _process_deriv_args(f, axis, x, delta)
    n = len(f.shape)
    #print('uwins.shape :',f.shape)
    axis_shape = [1]*n
    axis_shape[axis] = delta.size()[0]
    delta = delta.reshape(*axis_shape)
    #take = make_take(n_dims, slice_dim)
    take = make_take(n, axis)

    # First handle centered case
    slice0 = take(slice(None, -2))
    slice1 = take(slice(1, -1))
    slice2 = take(slice(2, None))
    delta_slice0 = take(slice(None, -1))
    delta_slice1 = take(slice(1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    delta_diff = delta[delta_slice1] - delta[delta_slice0]
    center = (- delta[delta_slice1] / (combined_delta * delta[delta_slice0]) * f[slice0]
              + delta_diff / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
              + delta[delta_slice0] / (combined_delta * delta[delta_slice1]) * f[slice2])

    # Fill in "left" edge with forward difference
    slice0 = take(slice(None, 1))
    slice1 = take(slice(1, 2))
    slice2 = take(slice(2, 3))
    delta_slice0 = take(slice(None, 1))
    delta_slice1 = take(slice(1, 2))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    big_delta = combined_delta + delta[delta_slice0]
    left = (- big_delta / (combined_delta * delta[delta_slice0]) * f[slice0]
            + combined_delta / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
            - delta[delta_slice0] / (combined_delta * delta[delta_slice1]) * f[slice2])

    # Now the "right" edge with backward difference
    slice0 = take(slice(-3, -2))
    slice1 = take(slice(-2, -1))
    slice2 = take(slice(-1, None))
    delta_slice0 = take(slice(-2, -1))
    delta_slice1 = take(slice(-1, None))

    combined_delta = delta[delta_slice0] + delta[delta_slice1]
    big_delta = combined_delta + delta[delta_slice1]
    right = (delta[delta_slice1] / (combined_delta * delta[delta_slice0]) * f[slice0]
             - combined_delta / (delta[delta_slice0] * delta[delta_slice1]) * f[slice1]
             + big_delta / (combined_delta * delta[delta_slice1]) * f[slice2])

    return torch.concat((left, center, right), axis=axis)

def divergence(uwind, vwind, delta_x, delta_y, parallel_scale, meridional_scale, dx_correction, dy_correction):
    du_dx = parallel_scale * first_derivative(uwind, axis=-1, delta=delta_x) - vwind * dx_correction
    dv_dy = meridional_scale * first_derivative(vwind, axis=-2, delta=delta_y) - uwind * dy_correction
    return du_dx + dv_dy

def vorticity(uwind, vwind, delta_x, delta_y, parallel_scale, meridional_scale, dx_correction, dy_correction):
    du_dy = meridional_scale * first_derivative(uwind, axis=-2, delta=delta_y) + vwind * dy_correction
    dv_dx = parallel_scale * first_derivative(vwind, axis=-1, delta=delta_x) + uwind * dx_correction
    return dv_dx - du_dy

def divergence_vorticity(uwind, vwind, delta_x, delta_y, parallel_scale, meridional_scale, dx_correction, dy_correction):
    d = divergence(uwind, vwind, delta_x, delta_y, parallel_scale, meridional_scale, dx_correction, dy_correction)
    v = vorticity(uwind, vwind, delta_x, delta_y, parallel_scale, meridional_scale, dx_correction, dy_correction)
    return d, v

class DivergenceVorticity(torch.nn.Module):
    def __init__(self, background, vars, var_means, var_stds, dv_parameter_file, device=None):
        super().__init__()

        device = device
        if not device:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.uwind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if 'u_component_of_wind' in var],
                                                    dtype = 'int32')).long().to(device)
        self.vwind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if 'v_component_of_wind' in var],
                                                    dtype='int32')).long().to(device)
        self.nowind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if \
                                                      'u_component_of_wind' not in var and \
                                                      'v_component_of_wind' not in var],
                                                    dtype='int32')).long().to(device)
        self.uwind_means = torch.from_numpy(
            np.array([var_means[var][0] for var in vars if 'u_component_of_wind' in var],
                     dtype = 'f4')).to(device)
        self.uwind_means = self.uwind_means.reshape(-1, 1, 1)
        self.vwind_means = torch.from_numpy(
            np.array([var_means[var][0] for var in vars if 'v_component_of_wind' in var],
                     dtype='f4')).to(device)
        self.vwind_means = self.vwind_means.reshape(-1, 1, 1)
        self.uwind_stds = torch.from_numpy(
            np.array([var_stds[var][0] for var in vars if 'u_component_of_wind' in var],
                     dtype='f4')).to(device)
        self.uwind_stds = self.uwind_stds.reshape(-1, 1, 1)
        self.vwind_stds = torch.from_numpy(
            np.array([var_stds[var][0] for var in vars if 'v_component_of_wind' in var],
                     dtype='f4')).to(device)
        self.vwind_stds = self.vwind_stds.reshape(-1, 1, 1)

        dv_f = h5py.File(dv_parameter_file, 'r')
        self.dx = torch.from_numpy(dv_f['delta_x'][:]).float().to(device)
        self.dy = torch.from_numpy(dv_f['delta_y'][:]).float().to(device)
        self.parallel_scale = torch.unsqueeze(torch.from_numpy(dv_f['parallel_scale'][:]).float(), 0).to(device)
        self.meridional_scale = torch.unsqueeze(torch.from_numpy(dv_f['meridional_scale'][:]).float(), 0).to(device)
        self.dx_correction = torch.unsqueeze(torch.from_numpy(dv_f['dx_correction'][:]).float(), 0).to(device)
        self.dy_correction = torch.unsqueeze(torch.from_numpy(dv_f['dy_correction'][:]).float(), 0).to(device)
        dv_f.close()

        self.sht = th.RealSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(device).float()
        self.inv_sht = th.InverseRealSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(device).float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2)).reshape(1, 1, -1).to(device)

    def forward(self, x, y):
        #print(x.get_device)
        #print(self.uwind_idxs.get_device)
        #print(self.uwind_stds.get_device)
        #print(self.uwind_means.get_device)

        uwind_unstand = x[self.uwind_idxs] * self.uwind_stds + self.uwind_means
        vwind_unstand = x[self.vwind_idxs] * self.vwind_stds + self.vwind_means
        divergence, vorticity = divergence_vorticity(uwind_unstand, vwind_unstand, self.dx, self.dy,
                                                     self.parallel_scale, self.meridional_scale, self.dx_correction,
                                                     self.dy_correction)
        return torch.concat((x[self.nowind_idxs], divergence, vorticity), axis = 0)

        #uwind_stand = x[self.uwind_idxs]
        #vwind_stand = x[self.vwind_idxs]
        #return torch.concat((x[self.nowind_idxs], uwind_stand, vwind_stand), axis = 0)

    def diffSHT(self, diff):
        sh_diff = self.sht(diff.to(self.device))
        return sh_diff

    def diffInvSHT(self, sht_diff):
        inv_sht_diff = self.inv_sht(sht_diff)
        return inv_sht_diff

class UVWwind(torch.nn.Module):
    def __init__(self, background, vars, device=None):
        super().__init__()

        self.device = device
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.uwind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if 'u_component_of_wind' in var],
                                                    dtype = 'int32')).long().to(self.device)
        self.vwind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if 'v_component_of_wind' in var],
                                                    dtype='int32')).long().to(self.device)
        self.nowind_idxs = torch.from_numpy(np.array([i for i, var in enumerate(vars) if \
                                                      'u_component_of_wind' not in var and \
                                                      'v_component_of_wind' not in var],
                                                    dtype='int32')).long().to(self.device)
        self.nowind_num = len(self.nowind_idxs)
        self.uwind_num = len(self.uwind_idxs)
        self.vwind_num = len(self.vwind_idxs)


        self.sht = th.RealSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(self.device).float()
        self.vec_sht = th.RealVectorSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(self.device).float()
        self.inv_sht = th.InverseRealSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(self.device).float()
        self.inv_vec_sht = th.InverseRealSHT(background.shape[-2], background.shape[-1], grid="equiangular").to(self.device).float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2)).reshape(1, 1, -1).to(self.device)

    def forward(self, x):
        uwind_stand = x[self.uwind_idxs]
        vwind_stand = x[self.vwind_idxs]
        return torch.concat((x[self.nowind_idxs], uwind_stand, vwind_stand), axis = 0)
    
    def diffSHT(self, diff):

        diff_scalar = diff[:self.nowind_num]
        diff_uwind = diff[self.nowind_num:self.nowind_num+self.uwind_num]
        diff_vwind = diff[-self.vwind_num:]

        diff_vector = torch.stack((diff_uwind,diff_vwind),dim=1)

        sht_diff_scalar = self.sht(diff_scalar.to(self.device))
        sht_diff_vector = self.vec_sht(diff_vector.to(self.device))

        sht_diff_uwind = sht_diff_vector[:,0]
        sht_diff_vwind = sht_diff_vector[:,1]
        sht_diff = torch.concatenate((sht_diff_scalar,
                                            sht_diff_uwind,
                                            sht_diff_vwind
                                            ),
                                            axis=0)
        return sht_diff
    
    def diffInvSHT(self, sht_diff):

        sht_diff_scalar = sht_diff[:self.nowind_num]
        sht_diff_uwind = sht_diff[self.nowind_num:self.nowind_num+self.uwind_num]
        sht_diff_vwind = sht_diff[-self.vwind_num:]

        sht_diff_vector = torch.stack((sht_diff_uwind,sht_diff_vwind),dim=1)

        inv_sht_diff_scalar = self.inv_sht(sht_diff_scalar)
        inv_sht_diff_vector = self.inv_vec_sht(sht_diff_vector)

        inv_sht_diff_uwind = inv_sht_diff_vector[:,0]
        inv_sht_diff_vwind = inv_sht_diff_vector[:,1]
        inv_sht_diff = torch.concatenate((inv_sht_diff_scalar,
                                            inv_sht_diff_uwind,
                                            inv_sht_diff_vwind
                                            ),
                                            axis=0)
        return inv_sht_diff