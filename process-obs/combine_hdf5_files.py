import os, sys, h5py
import numpy as np
from tqdm import tqdm



dir = os.path.join("D:\\argonne_data", "IGRA_v2.2_data-por_s19050404_e20230606_c20230606")
var_dict = {'press' : 'Pa',
            'press_col' : 2,
            'gph' : 'm',
            'temp' : 'degrees C to tenths',
            'rh' : 'percent to tenths',
            'dpdp' : 'degrees c to tenths',
            'uwind' : 'meters per second to tenths',
            'vwind' : 'meters per second to tenths'}

full_h5_file = 'irga_1415_surface_all.hdf5'
if os.path.exists(os.path.join(dir, full_h5_file)):
    os.remove(os.path.join(dir, full_h5_file))
h5_files = [file for file in os.listdir(dir) if '.hdf5' in file and 'surface' in file]
f_full = h5py.File(os.path.join(dir, full_h5_file), 'a')
for i, file in enumerate(h5_files):
    print()
    print('Processing file %d' % (i+1))
    print()
    f = h5py.File(os.path.join(dir, file), 'a')
    for year in list(f.keys()):
        if year not in f_full.keys():
            yr_grp = f_full.create_group(year)
        for month in list(f[year].keys()):
            if month not in f_full[year].keys():
                mth_grp = f_full[year].create_group(month)
            for day in list(f[year + '/' + month].keys()):
                if day not in f_full[year + '/' + month].keys():
                    day_grp = f_full[year + '/' + month].create_group(day)
                for hour in list(f[year + '/' + month + '/' + day].keys()):
                    if hour not in f_full[year + '/' + month + '/' + day].keys():
                        hr_grp = f_full[year + '/' + month + '/' + day].create_group(hour)
                    for var in list(f[year + '/' + month + '/' + day + '/' + hour].keys()):
                        print(year + '/' + month + '/' + day + '/' + hour + '/' + var)
                        if var not in f_full[year + '/' + month + '/' + day + '/' + hour].keys():
                            var_data = f_full[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                                var, f[year + '/' + month + '/' + day + '/' + hour + '/' + var].shape,
                                maxshape=(None, 4), dtype = 'i4'
                            )
                            var_data[:] = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                            var_data.attrs['press_units'] = var_dict['press']
                            var_data.attrs['press_col'] = var_dict['press_col']
                            var_data.attrs['units'] = var_dict[var]
                        else:
                            f_full[year + '/' + month + '/' + day + '/' + hour + '/' + var].resize(
                                f_full[year + '/' + month + '/' + day + '/' + hour + '/' + var].shape[0] +
                                f[year + '/' + month + '/' + day + '/' + hour + '/' + var].shape[0], 0
                            )
                            f_full[year + '/' + month + '/' + day + '/' + hour + '/' + var][
                                -f[year + '/' + month + '/' + day + '/' + hour + '/' + var].shape[0]:, :
                            ] = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
    f.close()
f_full.close()