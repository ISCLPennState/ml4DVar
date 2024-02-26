import re
import sys
from datetime import datetime
from datetime import timedelta
import copy
import xarray as xr
import cfgrib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import torch
import torch_harmonics as th
#matplotlib.use('TkAgg')

import json
import numpy as np
import os 

import h5py

import glob


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

vars_climax = [
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
    "vertical_velocity_50",
    "vertical_velocity_100",
    "vertical_velocity_150",
    "vertical_velocity_200",
    "vertical_velocity_250",
    "vertical_velocity_300",
    "vertical_velocity_400",
    "vertical_velocity_500",
    "vertical_velocity_600",
    "vertical_velocity_700",
    "vertical_velocity_850",
    "vertical_velocity_925",
    "vertical_velocity_1000", # unmeasurable
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

def get_forecast_h5(file):
    mode = 'r'
    print('file',file)
    f = h5py.File(file, mode)
    preds_36 = np.array(f['36'][:],dtype=np.double)
    preds_12 = np.array(f['12'][:],dtype=np.double)
    f.close()
    return preds_12, preds_36 

def get_forecast(file):
    f = open(file)
    data_j = json.load(f)
    variables = data_j['out_variables']
    preds_36 =  np.array(data_j['pred_norm'][-1][0])
    preds_12 = np.array(data_j['pred_norm'][1][0])
    print(np.shape(preds_12))
    #plt.imshow(preds[0,:,:])
    #plt.colorbar()
    #plt.show()
    return preds_12, preds_36

def get_var_list():
    f = open('/eagle/MDClimSim/mjp5595/data/0000.json')
    data_j = json.load(f)
    variables = data_j['out_variables']
    return variables

if __name__ == '__main__':
    print(vars)

    vars = vars_climax

    means_list = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds_list = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')

    for key in means_list.keys(): print(key)

    means = np.array([means_list[var] for var in vars])
    stds = np.array([stds_list[var] for var in vars])

    print(means)
    print(stds)
    print('np.shape(means)',np.shape(means))
    print('np.shape(means[0])',np.shape(means[0])) 

    date_idx = 0
    #data_dir = '/eagle/MDClimSim/mjp5595/data/stormer_val_forecasts/'
    data_dir = '/eagle/MDClimSim/mjp5595/data/climax/climax_val_forecasts/'
    #files = os.listdir(data_dir)
    files = glob.glob(data_dir+'????.h5')
    files.sort()

    print(files)
    print(len(files))
    files_full = files
    files = files[0:-1:4]

    max_days = len(files)
    print('max_days',max_days)

    sh_coeffs = np.zeros((max_days, len(vars), 128))
    hf_diff = np.zeros((max_days, len(vars), 128, 256))

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)

    for i,f_name in enumerate(files):
        print('\n', i, '\n')
        try:
            preds_12, preds_36 = get_forecast_h5(f_name)
            #preds_12, preds_36 = get_forecast(data_dir+f_name)
        except:
            print('skipping file due to h5 error',f_name)
            print('trying new file',files_full[i*4+1])
            #f_name = files_full[i*4+1]
            #preds_12, preds_36 = get_forecast(data_dir+f_name) 

        print('max min, mean',np.max(preds_36),np.min(preds_36),np.mean(preds_36))
        preds_36 = (preds_36 - means.reshape(-1, 1, 1))/stds.reshape(-1, 1, 1)
        preds_12 = (preds_12 - means.reshape(-1, 1, 1))/stds.reshape(-1, 1, 1)
        diff = preds_36 - preds_12
        diff = torch.from_numpy(diff)

        sh_diff = sht(diff)
        diff_hf = diff - inv_sht(sh_diff)
        sh_coeffs[i] = np.real(sh_diff[:, :, 0].cpu().numpy())
        hf_diff[i] = diff_hf.cpu().numpy()


    sh_var = np.var(sh_coeffs[:], axis = 0)
    hf_var = np.var(hf_diff[:], axis = 0)
    np.save('background_24hr_diff_sh_coeffs_var_climaxv2_standardized_128_uv.npy', sh_var)
    np.save('background_24hr_diff_hf_var_climaxv2_standardized_128_uv.npy', hf_var)
