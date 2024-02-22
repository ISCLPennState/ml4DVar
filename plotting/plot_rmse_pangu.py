import re, os, sys
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
from src.plotting import latitude_weighted_rmse, latitude_weighted_rmse_regions
from natsort import natsorted
import numpy as np
#matplotlib.use('TkAgg')

device = "cpu"

era5_filepath = '/eagle/MDClimSim/troyarcomano/ai-models/ERA_5/'

#run_str = '12hr_01012014_standardized_128' 
#run_str = '12hr_01012014_standardized_128_24hr_diff_hf_old'
#run_str = '12hr_01012014_standardized_24hr_diff_720'
#run_str = '12hr_01012014_standardized_128_12hr_both_b_inflate_100_uv_b_from_z'
run_str = '12hr_01012020_standardized_128_12hr_normal_troy_uv_inflate_both_b_100_scaled_but_no_q_obs_above_200_hpa_pangu_bug_fix_start_near_date'

dir = 'data/pangu'
analysis_files = natsorted([filename for filename in os.listdir(dir) if 'analysis' in filename\
        and f'{run_str}.npy' in filename])
#print(analysis_files)
background_files = natsorted([filename for filename in os.listdir(dir) if 'background' in filename\
        and f'{run_str}.npy' in filename])
#print(background_files)

vars = np.load('/eagle/MDClimSim/awikner/ml4dvar/var_list_pangu.npy')
var_dict = dict(zip(vars, np.arange(len(vars))))
analysis_rmse = np.zeros((len(analysis_files), len(vars)))
background_rmse = np.zeros((len(background_files), len(vars)))

analysis_rmse_nh = np.zeros((len(analysis_files), len(vars)))
background_rmse_nh = np.zeros((len(background_files), len(vars)))

analysis_rmse_sh = np.zeros((len(analysis_files), len(vars)))
background_rmse_sh = np.zeros((len(background_files), len(vars)))

means = np.load('/eagle/MDClimSim/awikner/ml4dvar/pangu_means.npy')
stds = np.load('/eagle/MDClimSim/awikner/ml4dvar/pangu_stds.npy')
lat = np.arange(-90,90.25,0.25)
lon = np.arange(0, 360, 0.25)

atmo_vars = ['z', 'q', 't', 'u_component_of_wind', 'v_component_of_wind']
atmo_vars_pangu = ['z', 'q', 't', 'u', 'v']
surface_vars = ['msl','u_component_of_wind_10m','v_component_of_wind_10m','t2m']
surface_vars_pangu = ['msl','u10','v10','t2m']

start_date = datetime(2020,1,1,0)
end_date = datetime(2020,1,1,0) + timedelta(hours = 12 * max(len(analysis_files), len(background_files)))

date_idx = 0
current_date = start_date
print(stds)
while current_date < end_date:
    print(current_date)
    ds_truth = xr.open_dataset(f'{era5_filepath}era5_data_{current_date.strftime("%Y%m%d%H")}00.nc')
    era5_data = np.zeros((len(vars), 721, 1440))
    plevels = ds_truth.isobaricInhPa.to_numpy()
    #print(plevels)
    for i, surface_var in enumerate(surface_vars_pangu):
        truth_in = ds_truth[surface_var].to_numpy().reshape(721, 1440)
        era5_data[i] = truth_in[::-1]
    for atmo_var, atmo_var_pangu in zip(atmo_vars, atmo_vars_pangu):
        for plevel in plevels:
            var_key = '%s_%dhPa' % (atmo_var, int(plevel))
            #print(var_dict[var_key], var_key)
            truth_in = ds_truth.sel(isobaricInhPa=plevel)[atmo_var_pangu].to_numpy().reshape(721, 1440)
            era5_data[var_dict[var_key]] = truth_in[::-1]
    era5_data = (era5_data - means.reshape(-1, 1, 1))/stds.reshape(-1, 1, 1)
    if date_idx < len(analysis_files):
        analysis = np.load(os.path.join(dir, analysis_files[date_idx]))[0]
        print('dir, analysis_files[date_idx]',dir, analysis_files[date_idx],date_idx,f'{era5_filepath}era5_data_{current_date.strftime("%Y%m%d%H")}00.nc')
        for i,var in enumerate(vars):
            analysis_rmse[date_idx, i] = latitude_weighted_rmse_regions(lat, true=era5_data[i],
                                       prediction=analysis[i], lat_axis=0)
            analysis_rmse_nh[date_idx, i] = latitude_weighted_rmse_regions(lat, true=era5_data[i],
                                       prediction=analysis[i], lat_axis=0,region='NH')
            analysis_rmse_sh[date_idx, i] = latitude_weighted_rmse_regions(lat, true=era5_data[i],
                                       prediction=analysis[i], lat_axis=0,region='SH')

            if var == 'z_500hPa':
                print('analysis mean,max,min',np.mean(analysis[i]*stds[i] + means[i]),np.max(analysis[i]*stds[i] + means[i]),np.min(analysis[i]*stds[i] + means[i]))
                print('era5 mean,max,min',np.mean(era5_data[i]*stds[i] + means[i]),np.max(era5_data[i]*stds[i] + means[i]),np.min(era5_data[i]*stds[i] + means[i])) 
                print('analysis_rmse[date_idx, i]',analysis_rmse[date_idx, i]*stds[i],date_idx)

    if date_idx < len(background_files):
        background = np.load(os.path.join(dir, background_files[date_idx]))[0]
        for i in range(len(vars)):
            background_rmse[date_idx, i] = latitude_weighted_rmse_regions(lat, true=era5_data[i],
                                       prediction=background[i], lat_axis=0)
    current_date = current_date + timedelta(hours = 12)
    date_idx += 1


print(analysis_rmse.shape)
print(background_rmse.shape)

for i, var in enumerate(vars):
    if var == 'z_500hPa':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots//pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 't_500hPa':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 'v_component_of_wind_200hPa':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 't2m':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 'msl':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 'q_1000hPa':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), analysis_rmse[:,i]*stds[i], 'x-', label = 'Analysis')
        plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), background_rmse[:,i]*stds[i], 'x-', label = 'Background')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()

for i, var in enumerate(vars):
    if var == 'z_500hPa':
        print(var)
        print(analysis_rmse[:,i]*stds[i])
        print(background_rmse[:,i]*stds[i])
        plt.plot(np.arange(0, 0.5*len(analysis_rmse_nh), 0.5), analysis_rmse_nh[:,i]*stds[i], 'x-', label = 'NH')
        plt.plot(np.arange(0, 0.5*len(analysis_rmse_sh), 0.5), analysis_rmse_sh[:,i]*stds[i], 'x-', label = 'SH')
        plt.xlabel('Days from Analysis Start')
        plt.ylabel('Lat-weighted RMSE')
        plt.title(f'Pangu Analysis for {var} from {start_date.strftime("%m/%d/%Y")}')
        plt.legend()
        plt.savefig(f'plots/pangu_analysis_rmse_nh_sh_{var}_{run_str}.png', dpi = 400, bbox_inches = 'tight')
        #plt.show()
        plt.close()
    
'''
plt.plot(np.arange(0, 0.5*len(analysis_rmse), 0.5), np.sqrt(np.mean(analysis_rmse**2.0, axis = 1)), 'x-', label = 'Analysis')
plt.plot(np.arange(0, 0.5*len(background_rmse), 0.5), np.sqrt(np.mean(background_rmse**2.0, axis = 1)), 'x-', label = 'Background')
plt.xlabel('Days from Analysis Start')
plt.ylabel('Lat-weighted RMSE')
plt.title(f'Pangu Analysis from {start_date.strftime("%m/%d/%Y")}')
plt.legend()
plt.savefig(f'plots/pangu/pangu_analysis_rmse_{run_str}.png', dpi = 400, bbox_inches = 'tight')
plt.show()
'''
