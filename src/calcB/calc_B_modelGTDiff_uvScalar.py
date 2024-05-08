import torch
import torch_harmonics as th
import numpy as np
import os 
import h5py
import glob
import sys

sys.path.append('/eagle/MDClimSim/mjp5595/ml4dvar/')
from stormer.varsStormer import varsStormer
from stormer.stormer_utils_pangu import StormerWrapperPangu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_forecast_h5(forecast_dir,idx,hour_diff):
    mode = 'r'
    file = os.path.join(forecast_dir,'norm_{:0>4d}.h5'.format(idx))
    print('forecast file :',file)
    f = h5py.File(file, mode)
    preds = np.array(f[str(hour_diff)][:],dtype=np.double)
    f.close()
    return preds 

def load_era5(era5_dir,idx,hour_diff,year,vars_stormer):
    mode = 'r'
    file_num = idx + (hour_diff // 6)
    file = os.path.join(era5_dir,'{}_{:0>4d}.h5'.format(year,file_num))
    print('era5 file :',file)
    f = h5py.File(file, mode)

    data_np = np.zeros((len(vars_stormer),128,256))
    for i,var in enumerate(vars_stormer):
        data_np[i] = f['input/{}'.format(var)][:]
    return data_np

def norm_era5(data_np,stormer_wrapper):
    data_torch = torch.from_numpy(data_np)
    data_torch_norm = stormer_wrapper.inp_transform(data_torch)
    data_np_norm = data_torch_norm.numpy()
    return data_np_norm

def norm(x_np,stormer_wrapper):
    x_torch = torch.from_numpy(x_np)
    x_torch_denorm = stormer_wrapper.inp_transform(x_torch)
    return x_torch_denorm

if __name__ == '__main__':

    save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/'

    vars_stormer = varsStormer().vars_stormer
    uwind_idxs = varsStormer().uwind_idxs
    vwind_idxs = varsStormer().vwind_idxs
    nowind_idxs = varsStormer().nowind_idxs

    vars_stormer = varsStormer().vars_stormer
    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=None,
        base_lead_time=[6],
        )

    date_idx = 0
    #forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_forecasts_2017_norm/'
    forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_val_forecast_lt12_2017/'
    era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    # v_sht = th.RealVectorSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)

    files_norm = glob.glob(forecast_dir+'norm_????.h5')

    hour_diff = 12
    pred_start_idxs = [0,2]

    for psi in pred_start_idxs:

        sh_coeffs = []
        hf_coeffs = []
        for i in range(0,len(files_norm)):
            if i % 4 != psi:
                continue
            #print('hour_diff, i :',hour_diff,i)
            try:
                preds = get_forecast_h5(forecast_dir,i,hour_diff=hour_diff)
                ground_truth_raw = load_era5(era5_dir,i,hour_diff,2017,vars_stormer)
                ground_truth = norm_era5(ground_truth_raw,stormer_wrapper)
            except:
                print('skipping file due to h5 error',files_norm[i])
                continue

            diff = preds - ground_truth
            diff = torch.from_numpy(diff).to(device)

            diff_scalar = diff[nowind_idxs]
            diff_uwind_std = diff[uwind_idxs]
            diff_vwind_std = diff[vwind_idxs]
            diff_all = torch.concatenate((diff_scalar,
                                              diff_uwind_std,
                                              diff_vwind_std
                                              ),
                                              axis=0)

            sh_diff = sht(diff_all.to(device))
            inv_sh_diff = inv_sht(sh_diff)

            hf_diff = diff_all.to(device) - inv_sh_diff

            sh_coeffs.append( np.real(sh_diff[:, :, 0].cpu().numpy()) ) # (69,128,129)
            hf_coeffs.append( hf_diff.cpu().numpy() )

        sh_var_norm = np.var(sh_coeffs[:], axis = 0)
        hf_var_norm = np.var(hf_coeffs[:], axis = 0)
        print('sh_var_norm shape/min/max :',sh_var_norm.shape,np.min(sh_var_norm),np.max(sh_var_norm))
        print('hf_var_norm shape/min/max :',hf_var_norm.shape,np.min(hf_var_norm),np.max(hf_var_norm))
        np.save(os.path.join(save_dir,'sh_scalarUV_{}hrModelvsGTNMC_12hrlt_{:0>2d}.npy'.format(hour_diff,6*psi)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_scalarUV_{}hrModelvsGTNMC_12hrlt_{:0>2d}.npy'.format(hour_diff,6*psi)), hf_var_norm)