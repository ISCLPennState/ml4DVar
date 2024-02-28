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

    save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/'

    vars_stormer = varsStormer().vars_stormer
    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=None,
        base_lead_time=[6],
        )

    date_idx = 0
    forecast_dir = '/eagle/MDClimSim/mjp5595/data/stormer/stormer_forecasts_2017_norm/'
    era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'

    sht = th.RealSHT(128, 256, grid = 'equiangular').to(device)
    inv_sht = th.InverseRealSHT(128, 256, grid = 'equiangular').to(device)

    files_norm = glob.glob(forecast_dir+'norm_????.h5')
    files_norm = files_norm[0:-1:4]
    num_files = len(files_norm)

    #files = files[0:-1:4]
    hour_diffs = [6,12,24,72,192]

    for hour_diff in hour_diffs:

        sh_coeffs_norm = np.zeros((len(files_norm), len(vars_stormer), 128))
        hf_diff_norm = np.zeros((len(files_norm), len(vars_stormer), 128, 256))

        for i in range(0,int(num_files-(hour_diff//2)),4):
            print('hour_diff, i :',hour_diff,i)
            try:
                preds = get_forecast_h5(forecast_dir,i,hour_diff=hour_diff)
                ground_truth_raw = load_era5(era5_dir,i,hour_diff,2017,vars_stormer)
                ground_truth = norm_era5(ground_truth_raw,stormer_wrapper)
            except:
                print('skipping file due to h5 error',files_norm[i])
                continue

            diff_norm = preds - ground_truth
            diff_norm = torch.from_numpy(diff_norm)

            sh_diff_norm = sht(diff_norm)
            #print('sh_diff_norm.shape :',sh_diff_norm.shape)
            diff_hf_norm = diff_norm - inv_sht(sh_diff_norm)
            # TODO why only take the first index here??
            sh_coeffs_norm[i] = np.real(sh_diff_norm[:, :, 0].cpu().numpy())
            hf_diff_norm[i] = diff_hf_norm.cpu().numpy()

        sh_var_norm = np.var(sh_coeffs_norm[:], axis = 0)
        hf_var_norm = np.var(hf_diff_norm[:], axis = 0)
        np.save(os.path.join(save_dir,'sh_{}hr_stormer_vs_era5.npy'.format(hour_diff)), sh_var_norm)
        np.save(os.path.join(save_dir,'hf_{}hr_stormer_vs_era5.npy'.format(hour_diff)), hf_var_norm)