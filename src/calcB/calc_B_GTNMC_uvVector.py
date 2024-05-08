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

def get_forecast_h5(idx,hour_diff=24):
    mode = 'r'
    file = os.path.join(forecast_dir,'norm_{:0>4d}.h5'.format(idx))
    print('file',file)
    f = h5py.File(file, mode)
    preds_12 = np.array(f['12'][:],dtype=np.double)
    preds_36 = np.array(f[str(int(12)+int(hour_diff))][:],dtype=np.double)
    f.close()
    return preds_12, preds_36 

def norm(x_np,stormer_wrapper):
    x_torch = torch.from_numpy(x_np)
    x_torch_norm = stormer_wrapper.inp_transform(x_torch)
    return x_torch_norm

def denorm(x_np,stormer_wrapper):
    x_torch = torch.from_numpy(x_np)
    x_torch_norm = stormer_wrapper.inp_transform(x_torch)
    return x_torch_norm

if __name__ == '__main__':

    save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/'

    vars_stormer = varsStormer().vars_stormer
    uwind_idxs = varsStormer().uwind_idxs
    vwind_idxs = varsStormer().vwind_idxs
    nowind_idxs = varsStormer().nowind_idxs

    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=None,
        base_lead_time=[6],
        )

    uwind_stds = stormer_wrapper.normalize_std[uwind_idxs].reshape(-1,1,1)
    vwind_stds = stormer_wrapper.normalize_std[vwind_idxs].reshape(-1,1,1)
    print('uwind_stds.shape :',uwind_stds.shape)
    print('vwind_stds.shape :',vwind_stds.shape)
    print('uwind_stds :',uwind_stds)
    print('vwind_stds :',vwind_stds)

    date_idx = 0
    #forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_forecasts_2017_norm/'
    forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_val_forecast_lt12_2017/'
    era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    vec_sht = th.RealVectorSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)
    inv_vec_sht = th.InverseRealVectorSHT(128, 256, grid = 'equiangular').to(device)

    files_norm = glob.glob(forecast_dir+'norm_????.h5')

    hour_diff = 24
    pred_start_idxs = [0,2]

    for psi in pred_start_idxs:
        print('psi :',psi)
        sh_coeffs = []
        hf_coeffs = []

        for i in range(0,len(files_norm)):
            if i % 4 != psi:
                continue
            try:
                ground_truth_0_raw = load_era5(era5_dir,i,0,2017,vars_stormer)
                ground_truth_0 = norm_era5(ground_truth_0_raw,stormer_wrapper)
                ground_truth_24_raw = load_era5(era5_dir,i,24,2017,vars_stormer)
                ground_truth_24 = norm_era5(ground_truth_24_raw,stormer_wrapper)
            except:
                print('skipping file due to h5 error',files_norm[i])
                continue

            diff = ground_truth_24 - ground_truth_0
            diff = torch.from_numpy(diff)

            diff_scalar = diff[nowind_idxs]
            sh_diff_scalar = sht(diff_scalar.to(device))

            diff_uwind_std = diff[uwind_idxs]
            diff_vwind_std = diff[vwind_idxs]
            diff_uwind_unstd = diff_uwind_std * uwind_stds
            diff_vwind_unstd = diff_vwind_std * vwind_stds

            diff_vector_unstd = torch.stack((diff_uwind_unstd,diff_vwind_unstd),dim=1)
            sh_diff_vector_unstd = vec_sht(diff_vector_unstd.to(device))
            sh_diff_uwind_unstd = sh_diff_vector_unstd[:,0]
            sh_diff_vwind_unstd = sh_diff_vector_unstd[:,1]

            sh_diff = torch.concatenate((sh_diff_scalar,
                                              sh_diff_uwind_unstd,
                                              sh_diff_vwind_unstd
                                              ),
                                              axis=0)

            inv_sh_diff_scalar = inv_sht(sh_diff_scalar)

            inv_sh_diff_vector_unstd = inv_vec_sht(sh_diff_vector_unstd)
            inv_sh_diff_uwind_unstd = inv_sh_diff_vector_unstd[:,0]
            inv_sh_diff_vwind_unstd = inv_sh_diff_vector_unstd[:,1]
            inv_sh_diff_uwind_std = inv_sh_diff_uwind_unstd / torch.Tensor(uwind_stds).to(device)
            inv_sh_diff_vwind_std = inv_sh_diff_vwind_unstd / torch.Tensor(vwind_stds).to(device)

            hf_diff_scalar = diff_scalar.to(device) - inv_sh_diff_scalar
            hf_diff_uwind = diff_uwind_std.to(device) - inv_sh_diff_uwind_std
            hf_diff_vwind = diff_vwind_std.to(device) - inv_sh_diff_vwind_std

            hf_diff = torch.concatenate((hf_diff_scalar,
                                              hf_diff_uwind,
                                              hf_diff_vwind
                                              ),
                                              axis=0)

            sh_coeffs.append( np.real(sh_diff[:, :, 0].cpu().numpy()) ) # (69,128,129)
            hf_coeffs.append( hf_diff.cpu().numpy() )

        sh_var_norm = np.var(sh_coeffs[:], axis = 0)
        hf_var_norm = np.var(hf_coeffs[:], axis = 0)
        print('sh_var_norm shape/min/max :',sh_var_norm.shape,np.min(sh_var_norm),np.max(sh_var_norm))
        print('hf_var_norm shape/min/max :',hf_var_norm.shape,np.min(hf_var_norm),np.max(hf_var_norm))
        np.save(os.path.join(save_dir,'sh_vectorUV_{}hrGTNMC_{:0>2d}.npy'.format(hour_diff,6*psi)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_vectorUV_{}hrGTNMC_{:0>2d}.npy'.format(hour_diff,6*psi)), hf_var_norm)