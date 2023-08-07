import dask, os, sys, h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, current_process

def get_var(line, var, var_dict):
    return line[var_dict[var][0]:var_dict[var][1]]

def get_etime(line, data_dict):
    etime = get_var(line, 'ETIME', data_dict)
    if etime in ['-9999', '-8888']:
        return 0
    else:
        etime_int = int(etime)
        if etime_int < 60:
            return etime_int
        else:
            return int(etime[-2:]) + 60*int(etime[:3])

def get_press(line, data_dict):
    press = get_var(line, 'PRESS', data_dict)
    #pflag = get_var(line, 'PFLAG', data_dict)
    if press == '-9999':# or pflag not in ['A', 'B']:
        return False, 0
    else:
        return True, int(press)

def get_gph(line, data_dict):
    gph = get_var(line, 'GPH', data_dict)
    zflag = get_var(line, 'ZFLAG', data_dict)
    if gph in ['-9999', '-8888'] or zflag not in ['A', 'B']:
        return False, 0
    else:
        return True, int(gph)

def get_temp(line, data_dict):
    temp = get_var(line, 'TEMP', data_dict)
    tflag = get_var(line, 'TFLAG', data_dict)
    if temp in ['-9999', '-8888'] or tflag not in ['A', 'B']:
        return False, 0
    else:
        return True, int(temp)

def get_rh(line, data_dict):
    rh = get_var(line, 'RH', data_dict)
    if rh in ['-9999', '-8888']:
        return False, 0
    else:
        return True, int(rh)

def get_dpdp(line, data_dict):
    dpdp = get_var(line, 'DPDP', data_dict)
    if dpdp in ['-9999', '-8888']:
        return False, 0
    else:
        return True, int(dpdp)

def get_uv(line, data_dict):
    wdir = get_var(line, 'WDIR', data_dict)
    wspd = get_var(line, 'WSPD', data_dict)
    if wdir in ['-9999', '-8888'] or wspd in ['-9999', '-8888']:
        return False, 0, 0
    else:
        u = round(float(wspd)*np.sin(float(wdir)*np.pi/180))
        v = round(float(wspd)*np.cos(float(wdir)*np.pi/180))
        return True, u, v

def add_data(group, data, data_name, lat, lon, press):
    if data_name not in group.keys():
        gph_data = group.create_dataset(data_name, (1, 4), maxshape=(None, 4), dtype='i4')
        gph_data[0, :] = np.array([lat, lon, press, data], dtype = 'i4')
    else:
        group[data_name].resize(group[data_name].shape[0] + 1, 0)
        group[data_name][-1, :] = np.array([lat, lon, press, data], dtype = 'i4')
    return

def append_vars(group, line, lat, lon, data_dict):
    ifpress, press = get_press(line, data_dict)
    if ifpress:
        ifgph, gph = get_gph(line, data_dict)
        if ifgph:
            add_data(group, gph, 'gph', lat, lon, press)
        iftemp, temp = get_temp(line, data_dict)
        if iftemp:
            add_data(group, temp, 'temp', lat, lon, press)
        ifrh, rh = get_rh(line, data_dict)
        if ifrh:
            add_data(group, rh, 'rh', lat, lon, press)
        ifdpdp, dpdp = get_dpdp(line, data_dict)
        if ifdpdp:
            add_data(group, dpdp, 'dpdp', lat, lon, press)
        ifwnd, u, v = get_uv(line, data_dict)
        if ifwnd:
            add_data(group, u, 'uwind', lat, lon, press)
            add_data(group, v, 'vwind', lat, lon, press)
    return

def append_radiosonde(f, lines, header_dict, data_dict):
    year = get_var(lines[0], 'YEAR', header_dict)
    month = get_var(lines[0], 'MONTH', header_dict)
    day = get_var(lines[0], 'DAY', header_dict)
    hour = get_var(lines[0], 'HOUR', header_dict)
    if hour != '99':
        lat = int(get_var(lines[0], 'LAT', header_dict))
        lon = int(get_var(lines[0], 'LON', header_dict))
        if year not in f.keys():
            yr_grp = f.create_group(year)
        idx_str = year
        if month not in f[idx_str].keys():
            mth_grp = f[idx_str].create_group(month)
        idx_str = idx_str + '/' + month
        if day not in f[idx_str].keys():
            day_grp = f[idx_str].create_group(day)
        idx_str = idx_str + '/' + day
        if hour not in f[idx_str].keys():
            hr_group = f[idx_str].create_group(hour)
        idx_str = idx_str + '/' + hour
        for line in lines[1:]:
            append_vars(f[idx_str], line, lat, lon, data_dict)
    return

def append_dataset(f, filepath, header_dict, data_dict):
    with open(filepath) as file:
        lines = file.readlines()
        lines_processed = 0
        #with tqdm(total = len(lines)) as pbar:
        while lines_processed < len(lines):
            data_len = int(get_var(lines[lines_processed], 'NUMLEV', header_dict))
            append_radiosonde(f, lines[lines_processed:lines_processed+data_len+1], header_dict, data_dict)
            lines_processed += (data_len + 1)
            #pbar.update(data_len + 1)
    return

def append_hdf_file(file):
    dir = os.path.join("D:\\argonne_data", "IGRA_v2.2_data-y2d_s20210101_e20230608_c20230608")
    colspecs_head = [(0,1), (1, 12), (13, 17), (18, 20), (21,23),
                (24, 26), (27, 31), (32, 36), (37, 45),
                (46, 54), (55, 62), (63, 71)]
    header_name = ['HEADREC', 'ID', 'YEAR', 'MONTH', 'DAY', 'HOUR',
              'RELTIME', 'NUMLEV', 'P_SRC', 'NP_SRC', 'LAT', 'LON']
    header_dict = dict(zip(header_name, colspecs_head))
    colspecs_data = [(0,1), (1,2), (3,8), (9,15), (15, 16), (16, 21), (21, 22), (22, 27), (27, 28),
                     (28, 33), (34, 39), (40, 45), (46, 51)]

    data_name = ['LVLTYP1', 'LVLTYP2', 'ETIME', 'PRESS', 'PFLAG', 'GPH', 'ZFLAG', 'TEMP', 'TFLAG', 'RH',
                 'DPDP', 'WDIR', 'WSPD']
    data_dict = dict(zip(data_name, colspecs_data))
    process_idx = current_process()._identity[0]
    h5_filepath = os.path.join(dir, 'irga_y2d_%s.hdf5' % process_idx)
    f = h5py.File(h5_filepath, 'a')
    append_dataset(f, os.path.join(dir, file), header_dict, data_dict)
    f.close()
    return

def print_file(file):
    print(file)
    return


if __name__ == '__main__':
    dir = os.path.join("D:\\argonne_data", "IGRA_v2.2_data-y2d_s20210101_e20230608_c20230608")
    filenames = [file for file in os.listdir(dir) if '.txt' in file]
    hdf_files = [file for file in os.listdir(dir) if '.hdf5' in file]
    for file in hdf_files:
        os.remove(os.path.join(dir, file))

    with Pool(processes = 10) as p:
        with tqdm(total = len(filenames)) as pbar:
            for _ in p.imap_unordered(append_hdf_file, filenames):
                pbar.update()
