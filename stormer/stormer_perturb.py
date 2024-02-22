import os, sys
#sys.path.append("/eagle/MDClimSim/mjp5595/ClimaX-v2/src/climax")
from torch.utils.data import IterableDataset, DataLoader
import torch
import h5py
from datetime import datetime, timedelta
import numpy as np
from src.dv import *
from src.obs_cummulative import ObsDatasetCum, ObsError 
from ml4dvar.stormer.stormer_utils import StormerWrapper

#from src.era5_iterative_dataset import ERA5OneStepRandomizedDataset, ERA5MultiLeadtimeDataset
from stormer.models.hub.vit_adaln import ViTAdaLN
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from ml4dvar.stormer.stormer_utils import StormerWrapper

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    save_dir_name = 'stormer_perturb_forecasts'

    start_date = datetime(2014, 1, 1, hour=0)
    end_date = datetime(2015, 12, 31, hour=12)
    da_window = 12
    model_step = 6
    obs_freq = 3

    save_dir = '/eagle/MDClimSim/mjp5595/data/{}/'.format(save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized_360_2.hdf5"

    means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')

    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/background_init_stormer_norm.npy' # This is just to initialize the model background

    from ml4dvar.stormer.varsStormer import varsStormer
    vars_stormer = varsStormer().vars_stormer

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'pressure']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0, 100.]
    obs_perc_err = [False, False, False, False, False, False]
    obs_err = ObsError(vars_stormer, var_types, var_obs_err, obs_perc_err, stds)

    # 3d var
    # (2014,1,1,1,0)->(2014,1,1,12,0)

    # 4d var
    # make analysis @ because we need the prev 12 hrs
    # (2014,1,1,12,0)
    # se_obs0 - (2014,1,1,1,0)->(2014,1,1,6,0)
    # se_obs1 - (2014,1,1,7,0)->(2014,1,1,12,0)

    # in 3d var se_obs makes the assumption that all the obs happen at the analysis time
    # 4d var we optimizing trajectory instead of point in time
    obs_steps = 1
    obs_dataset = ObsDatasetCum(filepath, start_date, end_date, vars_stormer, 
                                obs_freq=obs_freq, da_window=da_window, 
                                obs_start_idx=0, obs_steps=obs_steps,
                                )
    obs_loader = DataLoader(obs_dataset, batch_size=1, num_workers=0)
    

    ckpt_pth = '/eagle/MDClimSim/tungnd/stormer/models/6_12_24_climax_large_2_True_delta_8/checkpoints/epoch_015.ckpt'
    net = ViTAdaLN(
        in_img_size=(128, 256),
        list_variables=vars_stormer,
        patch_size=2,
        embed_norm=True,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    )
    net.to(device)
    net.eval()
    stormer_wrapper = StormerWrapper(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=net,
        list_lead_time=[6],
        ckpt=ckpt_pth,
        device=device,
    )

    #pytorch_total_params = sum(p.numel() for p in stormer_wrapper.net.parameters())
    #pytorch_trainable_params = sum(p.numel() for p in stormer_wrapper.net.parameters() if p.requires_grad)
    #print('Total model parameters : {}'.format(pytorch_total_params))
    #print('Trainable model parameters : {}'.format(pytorch_trainable_params))

    print('background_file_np :',background_file_np)
    background_f = np.load(background_file_np, 'r')
    background = torch.from_numpy(background_f.copy())
    #background = background.to(torch.float32)
    print('background_f.shape :',background_f.shape)
    print('background.shape :',background.shape)
    print('mean/min/max background temperature :',np.mean(background_f[0][0]),np.min(background_f[0][0]),np.max(background_f[0][0]))

    ########################################################################################################
    ########################################################################################################
    def run_forecast(x,
                    num_model_steps,
                    stormer_wrapper=stormer_wrapper,
                    vars_stormer=vars_stormer,
                    device=device
                    ):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        norm_preds,_,_ = stormer_wrapper.eval_multi_step(
            x.to(device),
            vars_stormer,
            num_model_steps)
        norm_preds = norm_preds[-1]
        return norm_preds

    def run_forecasts(x,
                    noise_level=0.00,
                    forecast_steps=2,
                    save_dir=save_dir,
                    stormer_wrapper=stormer_wrapper,
                    vars_stormer=vars_stormer,
                    device=device,
                    ):

        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        norm_preds,raw_preds = stormer_wrapper.eval_multi_step(
            x.to(device),
            vars_stormer,
            steps=forecast_steps)

        hf_n = h5py.File(os.path.join(save_dir, 'norm_forecast_noise{}.h5'.format(noise_level)),'w')
        hf_r = h5py.File(os.path.join(save_dir, 'raw_forecast_noise{}.h5'.format(noise_level)),'w')
        # forecasts [#forecasts,(vars,lat,lon)]
        for i in range(len(norm_preds)):
            hf_n.create_dataset(str(i), data=norm_preds[i].detach().cpu().numpy())
            hf_r.create_dataset(str(i), data=raw_preds[i].detach().cpu().numpy())
        return

    def gen_diffs(diffs, all_obs, H_idxs_unraveled_r, H_idxs_unraveled_c, vars_stormer):
        for v in range(len(vars_stormer)):
            #print('all_obs[0,0,v,:] :',all_obs[0,0,v,:])
            #print('all_obs[0,0,v,:].shape :',all_obs[0,0,v,:].shape)
            print('min/max all_obs[0,0,{},:] : {}/{}'.format(v,np.min(all_obs[0,0,v,:]),np.max(all_obs[0,0,v,:])))
            for idx, (r,c) in enumerate(zip(H_idxs_unraveled_r[0,0,v],H_idxs_unraveled_c[0,0,v])):

                # extra observations to ignore are [0,0]
                if r==0 and c==0:
                    continue

                #############################################
                # for random locations
                #############################################
                #else:
                #    r = np.random.randint(0,128)
                #    c = np.random.randint(0,256)
                #############################################
                #############################################

                #print('r,c :',r,c)
                # Increases observation size to center + surrounding pixels (9pixels total)
                directions = [[0,0],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
                diff = 1
                if all_obs[0,0,v,idx] < 0:
                    diff = -1
                for [dr,dc] in directions:
                    r_new = r+dr
                    c_new = c+dc
                    #print('r_new,c_new :',r_new,c_new)
                    try:
                        diffs[0,v,r_new,c_new] = diff
                    except:
                        continue
        return diffs
    ########################################################################################################
    ########################################################################################################

    noise_levels = [0, 0.01, 0.1, 1, 10, 100, 1000]

    all_obs, H_idxs, H_obs, shapes, obs_latlon = next(iter(obs_loader))
    all_obs = all_obs.detach().cpu().numpy()
    #print('all_obs.shape :',all_obs.shape)
    H_idxs = H_idxs.detach().cpu().numpy()
    #print('H_idxs.shape :',H_idxs.shape)
    #print('H_idxs[:10] :',H_idxs[0,0,0,:10])
    H_idxs = H_idxs[:,:,:,::4]
    #print('H_idxs.shape :',H_idxs.shape)
    #print('H_idxs[:5] :',H_idxs[0,0,0,:5])

    diffs = np.zeros_like(background_f) # (1,1,82,128,256)
    H_idxs_unraveled_r, H_idxs_unraveled_c = np.unravel_index(H_idxs,(128,256)) # These hold the r,c of observation points
    #print('H_idxs_unraveled_r :',H_idxs_unraveled_r.shape)
    #print('min/max H_idxs_unraved_r/c',np.min(H_idxs_unraveled_r),np.max(H_idxs_unraveled_r),np.min(H_idxs_unraveled_c),np.max(H_idxs_unraveled_c))
    
    diffs = gen_diffs(diffs, all_obs, H_idxs_unraveled_r, H_idxs_unraveled_c, vars_stormer)

    print('min/max diffs :',np.min(diffs),np.max(diffs))
    np.save(os.path.join(save_dir,'diffs'),diffs)
    diffs = torch.from_numpy(diffs)

    for noise_level in noise_levels:
        print('noise_level :',noise_level)
        background_perturbed = background + noise_level*diffs
        background_perturbed = background_perturbed.to(torch.float32)
        #print('torch.mean/min/max background_perturbed :',torch.mean(background_perturbed),torch.min(background_perturbed),torch.max(background_perturbed))
        #print('torch.mean/min/max temperature_perturbed :',torch.mean(background_perturbed[0][0]),torch.min(background_perturbed[0][0]),torch.max(background_perturbed[0][0]))
        with torch.inference_mode():
            run_forecasts(background_perturbed,
                          noise_level=noise_level,
                          forecast_steps=40,
                          save_dir=save_dir,
                          stormer_wrapper=stormer_wrapper,
                          vars_stormer=vars_stormer,
                          device=device,
                          )
