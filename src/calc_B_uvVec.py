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
    x_torch_denorm = stormer_wrapper.inp_transform(x_torch)
    return x_torch_denorm

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

    date_idx = 0
    forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_forecasts_2017_norm/'

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    vec_sht = th.RealVectorSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)
    inv_vec_sht = th.InverseRealVectorSHT(128, 256, grid = 'equiangular').to(device)

    files_norm = glob.glob(forecast_dir+'norm_????.h5')

    hour_diff = 24
    pred_start_idxs = [0,2]

    for psi in pred_start_idxs:
        print('psi :',psi)
        sh_coeffs_norm = []
        hf_diff = []
        sh_coeffs_norm_rev = []
        hf_diff_rev = []

        for i in range(0,len(files_norm)):
            if i % 4 != psi:
                continue
            try:
                preds_12_norm, preds_36_norm = get_forecast_h5(i,hour_diff=hour_diff)
            except:
                print('skipping file due to h5 error',files_norm[i])
                continue

            diff = preds_36_norm - preds_12_norm
            diff = torch.from_numpy(diff)

            diff_scalar = diff[nowind_idxs]
            diff_uwind = diff[uwind_idxs]
            diff_vwind = diff[vwind_idxs]

            diff_vector = torch.stack((diff_uwind,diff_vwind),dim=1)
            #print('diff_vector.shape :',diff_vector.shape)

            sh_diff_scalar = sht(diff_scalar.to(device))
            sh_diff_vector = vec_sht(diff_vector.to(device))
            #print('sh_diff_scalar.shape :',sh_diff_scalar.shape)
            #print('sh_diff_vector.shape :',sh_diff_vector.shape)

            sh_diff_uwind = sh_diff_vector[:,0]
            #sh_diff_uwind = sh_diff_vector
            sh_diff_vwind = sh_diff_vector[:,1]
            sh_diff = torch.concatenate((sh_diff_scalar,
                                              sh_diff_uwind,
                                              sh_diff_vwind
                                              ),
                                              axis=0)
            #print('sh_diff.shape :',sh_diff.shape)

            inv_sh_diff_scalar = inv_sht(sh_diff_scalar)
            inv_sh_diff_vector = inv_vec_sht(sh_diff_vector)
            #print('inv_sh_diff_scalar.shape :',inv_sh_diff_scalar.shape)
            #print('inv_sh_diff_vector.shape :',inv_sh_diff_vector.shape)

            inv_sh_diff_uwind = inv_sh_diff_vector[:,0]
            inv_sh_diff_vwind = inv_sh_diff_vector[:,1]
            #print('inv_sh_diff_uwind.shape :',inv_sh_diff_uwind.shape)
            
            hf_diff_scalar = diff_scalar.to(device) - inv_sh_diff_scalar
            hf_diff_uwind = diff_uwind.to(device) - inv_sh_diff_uwind
            hf_diff_vwind = diff_vwind.to(device) - inv_sh_diff_vwind

            hf_diff_combined = torch.concatenate((hf_diff_scalar,
                                              hf_diff_uwind,
                                              hf_diff_vwind
                                              ),
                                              axis=0)

            sh_coeffs_norm.append( np.real(sh_diff[:, :, 0].cpu().numpy()) ) # (69,128,129)
            hf_diff.append( hf_diff_combined.cpu().numpy() )

        sh_var_norm = np.var(sh_coeffs_norm[:], axis = 0)
        hf_var_norm = np.var(hf_diff[:], axis = 0)
        np.save(os.path.join(save_dir,'sh_stormer_{}hrPred_{}hrNMC_{:0>2d}.npy'.format(12,hour_diff,6*psi)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_stormer_{}hrPred_{}hrNMC_{:0>2d}.npy'.format(12,hour_diff,6*psi)), hf_var_norm)