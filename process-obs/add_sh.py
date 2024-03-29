import os, sys, h5py
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def compute_real_latlon(latlon):
    latlon_sign = np.sign(latlon)
    fl_latlon   = np.abs(float(latlon))
    deg_latlon  = np.floor(fl_latlon/10000)
    min_latlon  = np.floor((fl_latlon/100 - deg_latlon*100))
    sec_latlon  = fl_latlon - deg_latlon*10000 - min_latlon*100
    real_latlon = latlon_sign * (deg_latlon + min_latlon/60 + sec_latlon/(60**2))
    return real_latlon

def convert_var_tenths(var):
    return float(var)/10.

def compute_q(dpdp, temp, press):
    Td = temp - dpdp
    e = 6.112 * np.exp((17.67 * Td) / (Td + 243.5))
    q = (0.622 * e) / (press - (0.378 * e))
    return q

def get_dpdp_temp_overlap(dpdp_data, temp_data):
    dpdp_match = np.empty((0, 4)); temp_match = np.empty((0, 4))
    for i, coord in enumerate(dpdp_data[:, :3]):
        if_match = np.all(coord == temp_data[:, :3], axis = 1)
        if np.any(if_match):
            dpdp_match = np.vstack((dpdp_match, np.append(coord, dpdp_data[i,3])))
            temp_match = np.vstack((temp_match, np.append(coord, temp_data[if_match, 3])))
    return dpdp_data, temp_match

dir = "/eagle/MDClimSim/awikner"
h5_file = 'irga_y2d_all_2.hdf5'
proc_h5_file = 'irga_y2d_proc.hdf5'

f = h5py.File(os.path.join(dir, h5_file), 'r')
f_proc = h5py.File(os.path.join(dir, proc_h5_file), 'a')
hour_count = np.zeros((24, 6), dtype='i8')
hour_strs = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
             '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
hour_idxs = np.arange(24)
var_strs = ['dpdp','gph','rh', 'temp', 'uwind', 'vwind']
var_idxs = np.arange(6)
hour_dict = dict(zip(hour_strs, hour_idxs))
var_dict = dict(zip(var_strs, var_idxs))

var_dict = {'press' : 'Pa',
            'press_col' : 2,
            'gph' : 'm',
            'temp' : 'degrees c',
            'rh' : 'percent',
            'dpdp' : 'degrees c',
            'uwind' : 'meters per second',
            'vwind' : 'meters per second',
            'q' : 'grams per kilogram'}

pred_plevels = np.array([250, 500, 700, 850, 925], dtype='i4')*1000

for year in list(f.keys()):
    yr_grp = f_pred.create_group(year)
    for month in list(f[year].keys()):
        mth_grp = f_pred[year].create_group(month)
        for day in list(f[year + '/' + month].keys()):
            day_grp = f_pred[year + '/' + month].create_group(day)
            for hour in list(f[year + '/' + month + '/' + day].keys()):
                hr_group = f_pred[year + '/' + month + '/' + day].create_group(hour)
                for var in list(f[year + '/' + month + '/' + day + '/' + hour].keys()):
                    data = f[year + '/' + month + '/' + day + '/' + hour + '/' + var][:]
                    pred_plevel_idxs = [idx for idx in range(data.shape[0]) if data[idx, 2] in pred_plevels]
                    pred_plevel_data = data[pred_plevel_idxs]
                    real_data = f_pred[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                        var, data.shape, maxshape=(None, 4), dtype = 'f8'
                    )
                    real_data[:, 0] = compute_real_latlon(data[:, 0])
                    real_data[:, 1] = compute_real_latlon(data[:, 1])
                    real_data[:, 2] = float(data[:, 2])
                    if var == 'gph':
                        real_data[:, 3] = float(data[:, 3])
                    else:
                        real_data[:, 3] = convert_var_tenths(data[:, 3])
                    real_data.attrs['press_units'] = var_dict['press']
                    real_data.attrs['press_col'] = var_dict['press_col']
                    real_data.attrs['units'] = var_dict[var]
                dpdp_match, temp_match = get_dpdp_temp_overlap(\
                        f_pred[year + '/' + month + '/' + day + '/' + hour + '/dpdp'][:],
                                                               f_pred[year + '/' + month + '/' + day + '/' + hour + '/temp'][:])
                q_data = f_pred[year + '/' + month + '/' + day + '/' + hour].create_dataset(
                    'q', dpdp_match.shape, maxshape = (None, 4), dtype = 'f8'
                )
                q_data[:, :3] = dpdp_match[:, :3]
                q_data[:, 3] = compute_q(dpdp_match[:, 3], temp_match[:, 3], dpdp_match[:, 2])
                q_data.attrs['press_units'] = var_dict['press']
                q_data.attrs['press_col'] = var_dict['press_col']
                q_data.attrs['units'] = var_dict['q']
