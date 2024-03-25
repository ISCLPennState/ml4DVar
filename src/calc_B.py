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

def get_forecast_h5(file,hour_diff):
    mode = 'r'
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

    save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/'

    vars_stormer = varsStormer().vars_stormer
    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=None,
        base_lead_time=[6],
        )

    date_idx = 0
    data_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_forecasts_2017_norm/'

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)

    files_norm = glob.glob(data_dir+'norm_????.h5')
    files_raw = glob.glob(data_dir+'raw_????.h5')
    files_norm.sort()
    files_raw.sort()
    files_norm = files_norm[0:-1:4]
    files_raw = files_raw[0:-1:4]

    #files = files[0:-1:4]
    hour_diffs = [12,24,72,144,192]

    for hour_diff in hour_diffs:
        print('hour_diff :',hour_diff)
        sh_coeffs_norm = np.zeros((len(files_norm), len(vars_stormer), 128))
        hf_diff_norm = np.zeros((len(files_norm), len(vars_stormer), 128, 256))
        sh_coeffs_raw_norm = np.zeros((len(files_norm), len(vars_stormer), 128))
        hf_diff_raw_norm = np.zeros((len(files_norm), len(vars_stormer), 128, 256))

        for i in range(len(files_norm)):
            print('i :',i)
            try:
                preds_12_norm, preds_36_norm = get_forecast_h5(files_norm[i],hour_diff=hour_diff)
                preds_12_raw, preds_36_raw = get_forecast_h5(files_raw[i],hour_diff=hour_diff)
            except:
                print('skipping file due to h5 error',files_norm[i])
                print('skipping file due to h5 error',files_raw[i])
                continue
                #f_name = files_full[i*4+1]
                #preds_12, preds_36 = get_forecast(data_dir+f_name) 


            #diff_norm = preds_36_norm - preds_12_norm
            diff_norm = preds_12_norm - preds_36_norm
            diff_norm = torch.from_numpy(diff_norm)
            sh_diff_norm = sht(diff_norm)
            print('sh_diff_norm.shape :',sh_diff_norm.shape)
            diff_hf_norm = diff_norm - inv_sht(sh_diff_norm)
            sh_coeffs_norm[i] = np.real(sh_diff_norm[:, :, 0].cpu().numpy())
            hf_diff_norm[i] = diff_hf_norm.cpu().numpy()
            #
            #diff_raw_norm = norm(preds_36_raw,stormer_wrapper) - norm(preds_12_raw,stormer_wrapper)
            #sh_diff_raw_norm = sht(diff_raw_norm)
            #diff_hf_raw_norm = diff_raw_norm - inv_sht(sh_diff_raw_norm)
            #sh_coeffs_raw_norm[i] = np.real(sh_diff_raw_norm[:, :, 0].cpu().numpy())
            #hf_diff_raw_norm[i] = diff_hf_raw_norm.cpu().numpy()

        sh_var_norm = np.var(sh_coeffs_norm[:], axis = 0)
        hf_var_norm = np.var(hf_diff_norm[:], axis = 0)
        sh_var_raw_norm = np.var(sh_coeffs_raw_norm[:], axis = 0)
        hf_var_raw_norm = np.var(hf_diff_raw_norm[:], axis = 0)
        np.save(os.path.join(save_dir,'sh_{}hr_stormer_norm_NegB.npy'.format(hour_diff)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_{}hr_stormer_norm_NegB.npy'.format(hour_diff)), hf_var_norm)
        #np.save(os.path.join(save_dir,'sh_{}hr_stormer_raw_norm.npy'.format(hour_diff)), sh_var_raw_norm)
        #np.save(os.path.join(save_dir,'hf_{}hr_stormer_raw_norm.npy'.format(hour_diff)), hf_var_raw_norm)