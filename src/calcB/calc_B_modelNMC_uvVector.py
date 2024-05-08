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

        sh_coeffs_m0_complex = []

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
            sh_diff_scalar = sht(diff_scalar.to(device))

            diff_uwind_std = diff[uwind_idxs]
            diff_vwind_std = diff[vwind_idxs]
            diff_uwind_unstd = diff_uwind_std * uwind_stds
            diff_vwind_unstd = diff_vwind_std * vwind_stds
            #print(' diff_uwind_std min/max :',torch.min(diff_uwind_std),torch.max(diff_uwind_std))
            #print(' diff_vwind_std min/max :',torch.min(diff_vwind_std),torch.max(diff_vwind_std))
            #print(' diff_uwind_unstd min/max :',torch.min(diff_uwind_unstd),torch.max(diff_uwind_unstd))
            #print(' diff_vwind_unstd min/max :',torch.min(diff_vwind_unstd),torch.max(diff_vwind_unstd))

            #diff_vector = torch.stack((diff_uwind,diff_vwind),dim=1)
            diff_vector_unstd = torch.stack((diff_uwind_unstd,diff_vwind_unstd),dim=1)
            sh_diff_vector_unstd = vec_sht(diff_vector_unstd.to(device))
            sh_diff_uwind_unstd = sh_diff_vector_unstd[:,0]
            sh_diff_vwind_unstd = sh_diff_vector_unstd[:,1]
            #sh_diff_uwind_std = sh_diff_uwind_unstd / torch.Tensor(uwind_stds).to(device)
            #sh_diff_vwind_std = sh_diff_vwind_unstd / torch.Tensor(vwind_stds).to(device)
            #print(' sh_diff_uwind_unstd min/max :',torch.min(torch.real(sh_diff_uwind_unstd)),torch.max(torch.real(sh_diff_uwind_unstd)))
            #print(' sh_diff_vwind_unstd min/max :',torch.min(torch.real(sh_diff_vwind_unstd)),torch.max(torch.real(sh_diff_vwind_unstd)))
            #print(' sh_diff_uwind_std min/max :',torch.min(torch.real(sh_diff_uwind_std)),torch.max(torch.real(sh_diff_uwind_std)))
            #print(' sh_diff_vwind_std min/max :',torch.min(torch.real(sh_diff_vwind_std)),torch.max(torch.real(sh_diff_vwind_std)))
            #print('diff_vector.shape :',diff_vector.shape)
            #print('sh_diff_scalar.shape :',sh_diff_scalar.shape)
            #print('sh_diff_vector.shape :',sh_diff_vector.shape)

            sh_diff = torch.concatenate((sh_diff_scalar,
                                              sh_diff_uwind_unstd,
                                              sh_diff_vwind_unstd
                                              ),
                                              axis=0)
            print(' sh_diff shape/min/max :',sh_diff.shape,torch.min(torch.real(sh_diff)),torch.max(torch.real(sh_diff)))
            print(' sh_diff[:,:,0] :',sh_diff[:,:,0])
            #print('sh_diff.shape :',sh_diff.shape)

            inv_sh_diff_scalar = inv_sht(sh_diff_scalar)

            inv_sh_diff_vector_unstd = inv_vec_sht(sh_diff_vector_unstd)
            inv_sh_diff_uwind_unstd = inv_sh_diff_vector_unstd[:,0]
            inv_sh_diff_vwind_unstd = inv_sh_diff_vector_unstd[:,1]
            inv_sh_diff_uwind_std = inv_sh_diff_uwind_unstd / torch.Tensor(uwind_stds).to(device)
            inv_sh_diff_vwind_std = inv_sh_diff_vwind_unstd / torch.Tensor(vwind_stds).to(device)
            #print('inv_sh_diff_scalar.shape :',inv_sh_diff_scalar.shape)
            #print('inv_sh_diff_vector.shape :',inv_sh_diff_vector.shape)
            #print('inv_sh_diff_uwind.shape :',inv_sh_diff_uwind.shape)
            
            hf_diff_scalar = diff_scalar.to(device) - inv_sh_diff_scalar
            hf_diff_uwind = diff_uwind_std.to(device) - inv_sh_diff_uwind_std
            hf_diff_vwind = diff_vwind_std.to(device) - inv_sh_diff_vwind_std

            hf_diff_combined = torch.concatenate((hf_diff_scalar,
                                              hf_diff_uwind,
                                              hf_diff_vwind
                                              ),
                                              axis=0)

            sh_diff0 = sh_diff[:,:,0]
            print(' sh_diff[:,:,0] shape/min/max :',sh_diff0.shape,torch.min(torch.real(sh_diff0)),torch.max(torch.real(sh_diff0)))
            sh_coeffs_m0_complex.append(sh_diff0.cpu().numpy())

            sh_coeffs_norm.append( np.real(sh_diff[:, :, 0].cpu().numpy()) ) # (69,128,129)
            hf_diff.append( hf_diff_combined.cpu().numpy() )


        np.save(os.path.join(save_dir,'sh_coeffs_m0_complex.npy'),np.array(sh_coeffs_m0_complex))
        np.save(os.path.join(save_dir,'hf_diff.npy'),np.array(hf_diff))

        sh_var_norm = np.var(sh_coeffs_norm[:], axis = 0)
        hf_var_norm = np.var(hf_diff[:], axis = 0)
        print('sh_var_norm shape/min/max :',sh_var_norm.shape,np.min(sh_var_norm),np.max(sh_var_norm))
        print('hf_var_norm shape/min/max :',hf_var_norm.shape,np.min(hf_var_norm),np.max(hf_var_norm))
        np.save(os.path.join(save_dir,'sh_stormer_uvVector_{}hrPred_{}hrNMC_12hrlt_{:0>2d}.npy'.format(12,hour_diff,6*psi)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_stormer_uvVector_{}hrPred_{}hrNMC_12hrlt_{:0>2d}.npy'.format(12,hour_diff,6*psi)), hf_var_norm)