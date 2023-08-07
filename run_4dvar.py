import os, sys
sys.path.append("/eagle/MDClimSim/awikner/climax_4dvar_troy")
from torch.utils.data import IterableDataset, DataLoader
import torch
#from d2l import torch as d2l
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
from src.dv import *
import time

from climax.global_forecast_4dvar.train import main

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)

NUM_MODEL_STEPS=2 #Current climax works at 6 hours so with assimilation wind of 12 hours we need to call the model twice

if __name__ == '__main__':
    filepath = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"
    means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
    dv_param_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'
    background_err_file = '/eagle/MDClimSim/awikner/background_err_sh_coeffs_var.npy'
    background_err_hf_file = '/eagle/MDClimSim/awikner/background_err_hf_var.npy'
    background_file = '/eagle/MDClimSim/troyarcomano/ClimaX/predictions_test/forecasts.hdf5'
    # filepath = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\irga_1415_test1_obs.hdf5'
    # means = np.load('C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\normalize_mean.npz')
    # stds = np.load('C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\normalize_std.npz')
    # dv_param_file = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\dv_params_128_256.hdf5'
    # background_err_file = 'C:\\Users\\user\\Dropbox\\AlexanderWikner_1\\UMD_Grad_School\\aieada\\climaX_4dvar\\background_err_sh_coeffs_std.npy'

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

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0]
    obs_perc_err = [False, False, False, False, False]
    obs_err = ObsError(vars, var_types, var_obs_err, obs_perc_err, stds)
    # print(obs_err.obs_err)
    dv_layer = DivergenceVorticity(vars, means, stds, dv_param_file)

    background_err = torch.from_numpy(np.load(background_err_file)).float()
    background_err = background_err[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]
    background_err_hf = torch.from_numpy(np.load(background_err_hf_file)).float()
    background_err_hf = background_err_hf[
        torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]

    start_date = datetime(2014, 1, 1, hour=0)
    end_date = datetime(2015, 12, 31, hour=12)
    window_len = 0
    window_step = 12
    model_step = 12

    obs_dataset = ObsDataset(filepath, start_date, end_date, window_len, window_step, model_step, vars)

    loader = DataLoader(obs_dataset, batch_size=1, num_workers=0)

    nn_model = main()  # torch.nn.Identity() #main()

    nn_model.to(device)
    nn_model.eval()

    nn_model.full_input = torch.clone(nn_model.full_input).to(device)
    nn_model.lead_times = torch.clone(nn_model.lead_times).to(device)

    background_f = h5py.File(background_file, 'r')
    start_idx = 0
    background = torch.unsqueeze(torch.from_numpy(background_f['truth_12hr'][start_idx]), 0)
    background_f.close()

    fourd_da = FourDVar(nn_model, loader, background, background_err, background_err_hf, obs_err, dv_layer)
    fourd_da.fourDvar(forecast=True)