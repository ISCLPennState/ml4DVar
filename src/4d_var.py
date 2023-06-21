from torch.utils.data import IterableDataset, DataLoader
import torch
from d2l import torch as d2l
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
import time

def observe_linear(x, H_idxs, H_vals):
    output = torch.sum(H_vals * torch.concat((x[H_idxs[0]], x[H_idxs[1]], x[H_idxs[2]], x[H_idxs[3]]), axis = 1), axis = 1)
    return output
def fourDresidual(x, *args):
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
        sht_scaler - tensor for scaling m > 0 SHT coefficients
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
    sht_scaler = args[9]

    num_vars = x.shape[0]
    time_steps = obs.shape[0]
    #total_obs = torch.sum(num_obs)

    # Compute background error with identity background error covariance
    coeff_diff = sht(x - background)
    se_background = torch.sum(torch.abs(coeff_diff * torch.conj(coeff_diff))  * sht_scaler.reshape(1, -1)) / background_err

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
class ObsDataset(IterableDataset):
    def __init__(self, file_path, start_datetime, end_datetime, window_len, window_step, model_step, vars):
        super(ObsDataset).__init__()
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
            shapes = np.zeros((self.window_len_idxs, len(self.vars)), dtype = int)
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                shapes[i, j] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var].shape[0]
            max_obs = np.max(shapes)
            all_obs = np.zeros((self.window_len_idxs, len(self.vars), max_obs))
            H_idxs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs), dtype = 'i4')
            H_obs = np.zeros((self.window_len_idxs, len(self.vars), 4*max_obs))
            for (i, obs_datetime), (j, var) in product(enumerate(obs_datetimes), enumerate(self.vars)):
                all_obs[i, j, :shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var][:, 2]
                H_idxs[i, j, :4*shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 0]
                H_obs[i, j, :4 * shapes[i, j]] = f[obs_datetime.strftime("%Y/%m/%d/%H") + '/' + var + '_H'][:, 1]
            output = (all_obs, H_idxs, H_obs, shapes)
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

class FourDVar():
    def __init__(self, nn_model, obs_dataloader, background, background_err = 0.1, obs_err = 0.1, lr = 0.1,
                 max_lbfgs_iter = 100):
        super(FourDVar).__init__()
        self.save_hyperparamters()
        #self.board = d2l.ProgressBoard(xlabel = '4D Var loss', ylabel = 'Cycle Iteration',
        #                               xlim = [0, obs_dataloader.num_cycles])
        self.sht = th.RealSHT(background.shape[2], background.shape[3], grid="equiangular").to('cpu').float()
        self.sht_scaler = torch.from_numpy(np.append(1., np.ones(self.sht.mmax - 1)*2))

    def loss(self):
        return fourDresidual(self.x[0],
                             self.background[0],
                             self.background_err,
                             self.all_obs[0],
                             self.H_idxs[0],
                             self.H_obs[0],
                             self.n_obs[0],
                             self.obs_err,
                             self.nn_model,
                             self.sht,
                             self.sht_scaler)

    def configure_optimizer(self):
        return torch.optim.LBFGS([self.x], lr = self.lr, max_iter = self.max_lbfgs_iter)

    def cycle(self):
        def closure():
            self.optim.zero_grad()
            loss = self.loss()
            loss.backward()
            return loss
        self.optim.step(closure)
        cycle_loss = self.optim.state_dict()['state'][0]['prev_loss']
        for i in range(self.obs_dataloader.window_step_idxs):
            self.x = self.nn_model(self.x.detach())
        self.background = self.x.detach()
        self.x.requires_grad_(True)
        return self.background, cycle_loss
        #self.plot('loss', cycle_loss, False)

    def fourDvar(self):
        self.x = self.background
        self.x.requires_grad_(True)
        for self.all_obs, self.H_idxs, self.H_obs, self.n_obs in self.obs_dataloader:
            self.optim = self.configure_optimizer()
            _, cycle_loss = self.cycle()
            print('Cycle loss: %0.2f' % cycle_loss)

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in local_vars.items()
                        if k not in set(ignore + ['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)


if __name__ == '__main__':
    filepath = "C://Users//user//Dropbox//AlexanderWikner_1//UMD_Grad_School//aieada//process_obs//irga_1415_test1_obs.hdf5"
    start_date = datetime(2014, 1, 1, hour = 0)
    end_date = datetime(2015, 12, 31, hour = 12)
    window_len = 24
    window_step = 12
    model_step = 12

    var_types = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
    gp_levels = [500, 700, 850, 925]
    levels = [250, 500, 700, 850, 925]
    vars = []
    for var_type in var_types:
        if var_type == 'geopotential':
            plevels = gp_levels
        else:
            plevels = levels
        for level in plevels:
            vars.append('%s_%d' % (var_type, level))

    obs_dataset = ObsDataset(filepath, start_date, end_date, window_len, window_step, model_step, vars)

    loader = DataLoader(obs_dataset, batch_size = 1, num_workers=0)

    fourd_da = FourDVar(nn_model, loader, background)

    """
    tic = time.perf_counter()
    iters = 0
    num_obs = 0
    for batch in loader:
        iters += 1
        num_obs += torch.sum(batch[3])
    toc = time.perf_counter()
    avg_obs = num_obs/(3 * iters)
    print('Total time: %f sec., Total iters: %d' % ((toc - tic), iters))
    print('Avg. # of Obs. per 12 hours: %0.2f' % avg_obs)
    print('Load time per batch: %f sec.' % ((toc - tic)/iters))
    """


