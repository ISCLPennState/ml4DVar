import h5py
import os
import sys
import numpy as np
import torch

# Create dense ERA5 observation dataset w/ parallelism
# sys.argv[1] will be the var_idx to use

sys.path.append('/eagle/MDClimSim/mjp5595/ml4dvar/stormer/')
from stormer_utils_pangu import StormerWrapperPangu
from varsStormer import varsStormer

sys.path.append('/eagle/MDClimSim/mjp5595/ml4dvar/')
sys.path.append('/eagle/MDClimSim/mjp5595/ml4dvar/src')
from obs_cummulative import *

def read_era5(data,vars_stormer):
    data_np = np.zeros((len(vars_stormer),128,256))
    for i,var in enumerate(vars_stormer):
        data_np[i] = data['input/{}'.format(var)][:]
    return data_np

vars_stormer = varsStormer().vars_stormer
vars_units = varsStormer().var_units

irga_obs = h5py.File("/eagle/MDClimSim/mjp5595/ml4dvar/obs/igra_141520_stormer_obs_standardized_360_3.hdf5")

era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'
era5_2020_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/test/'

lats = np.linspace(-90,90,128)
lons = np.linspace(0,360,256)

means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')

#era5_obs = h5py.File('/eagle/MDClimSim/mjp5595/ml4dvar/data/era5_obs.h5', 'a')
#with h5py.File('/eagle/MDClimSim/mjp5595/ml4dvar/obs/era5_dense_obs.h5', 'w') as era5_obs:
with h5py.File('/eagle/MDClimSim/mjp5595/ml4dvar/obs/era5_dense_grid_2014-{}.h5'.format(int(sys.argv[1])), 'w') as era5_obs:
    for year in ['2014','2015','2020']:
        yr_grp = era5_obs.require_group(year)
        year_hr_idx = 0
        for month in list(irga_obs['{}'.format(year)].keys()):
            mth_grp = era5_obs['{}'.format(year)].require_group(str(month))
            for day in list(irga_obs['{}/{}'.format(year,month)].keys()):
                day_grp = era5_obs['{}/{}'.format(year,month)].require_group(day)
                for hour in list(irga_obs['{}/{}/{}'.format(year,month,day)].keys()):
                    if int(hour) % 6 != 0:
                        continue

                    if year != '2014' or (str(month) != '01' and str(month) != '02' and str(month) != '03'):
                        continue

                    print('[{}] - {}/{}/{}/{} - {:0>4d}'.format(sys.argv[1],year,month,day,hour,year_hr_idx))
                    hr_group = era5_obs['{}/{}/{}'.format(year,month,day)].require_group(hour)

                    # load era5
                    if year == '2014' or year == '2015':
                        try:
                            era5_data = torch.from_numpy(read_era5(h5py.File(os.path.join(era5_dir,'{}_{:0>4d}.h5'.format(year,year_hr_idx))),vars_stormer))
                        except:
                            continue
                    else:
                        try:
                            era5_data = torch.from_numpy(read_era5(h5py.File(os.path.join(era5_2020_dir,'{}_{:0>4d}.h5'.format(year,year_hr_idx))),vars_stormer))
                        except:
                            continue
                    year_hr_idx += 1

                    for var_idx,var in enumerate(vars_stormer):
                        if var_idx != int(sys.argv[1]):
                            continue

                        era5_var_data = []
                        era5_var_data_H = []
                        for row in range(128):
                            for col in range(256):
                                era5_var_loc_data = era5_data[var_idx,row,col]

                                era5_var_loc_data_lat = (row+1)*(180./128.) - 90.
                                era5_var_loc_data_lon = (col+1)*(360./256.)
                                era5_var_data.append([era5_var_loc_data_lat,
                                                      era5_var_loc_data_lon,
                                                      era5_var_loc_data])

                                era5_var_loc_data_H_idx = np.zeros(4)
                                r1, c1 = row, col
                                r2 = r1 + 1 if r1 < 127 else r1
                                c2 = c1 + 1 if c1 < 255 else c1
                                era5_var_loc_data_H_idx[0] = np.ravel_multi_index([r1,c1],(128,256))
                                era5_var_loc_data_H_idx[1] = np.ravel_multi_index([r2,c1],(128,256))
                                era5_var_loc_data_H_idx[2] = np.ravel_multi_index([r1,c2],(128,256))
                                era5_var_loc_data_H_idx[3] = np.ravel_multi_index([r2,c2],(128,256))
                                era5_var_loc_data_H_obs = np.array([1,0,0,0])
                                era5_var_data_H.extend(np.stack((era5_var_loc_data_H_idx,
                                                                era5_var_loc_data_H_obs),
                                                                axis=1)
                                )

                                #print()
                                #print('col :',col)
                                #print('era5_var_data :',era5_var_data)
                                #print('era5_var_data_H[-4:] :',era5_var_data_H[-4:])

                        era5_var_data = np.array(era5_var_data)
                        era5_var_data_H = np.array(era5_var_data_H)
                        #print('era5_var_data.shape :',era5_var_data.shape)
                        #print('era5_var_data_H.shape :',era5_var_data_H.shape)
                        era5_var_data[:,2] = (era5_var_data[:,2] - means[var]) / stds[var]

                        #print()
                        #print('era5_var_data :',era5_var_data)
                        #print('era5_var_data_H :',era5_var_data_H)

                        era5_obs_dataset = era5_obs['{}/{}/{}/{}'.format(year,month,day,hour)].create_dataset(
                            '{}'.format(var), data=era5_var_data, dtype = 'f8'
                        )
                        era5_obs_dataset_H = era5_obs['{}/{}/{}/{}'.format(year,month,day,hour)].create_dataset(
                            '{}_H'.format(var), data=era5_var_data_H, dtype = 'f8'
                        )