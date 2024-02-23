import os, sys
sys.path.append("/eagle/MDClimSim/mjp5595/ClimaX-v2/src/climax")
from torch.utils.data import IterableDataset, DataLoader
import torch
#from d2l import torch as d2l
import inspect
import h5py
from datetime import datetime, timedelta
import numpy as np
from itertools import product
import torch_harmonics as th
from src.dv import *
#from src.obs import ObsDataset, ObsError 
from src.obs_cummulative import ObsDatasetCum, ObsError 
#from src.var_4d import FourDVar
from src.var_4d_reformatted import FourDVar
import time
from climax_utils import ClimaXWrapper

from arch_swin import ClimaXSwin
from arch import ClimaX
#from ../Climax-v2/src/climax/arch_swin import ClimaXSwin

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)
import logging


if __name__ == '__main__':

    save_dir_name = 'climaX_defVars_noNoise'

    start_date = datetime(2014, 1, 1, hour=0)
    end_date = datetime(2015, 12, 31, hour=12)
    da_window = 12
    model_step = 6
    obs_freq = 3

    save_dir = '/eagle/MDClimSim/mjp5595/data/{}/'.format(save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #filepath = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5" # Old Observations
    #filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized.hdf5"
    #filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized_360.hdf5"
    filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized_360_2.hdf5"

    means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')

    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/background_starter.npy' # This is just to initialize the model background
    pretrained_path = "/eagle/MDClimSim/tungnd/ClimaX/exps/global_forecast_climax/dinov2_vitl14_iterative_predict_diff_4_steps/checkpoints/epoch_018.ckpt"
    #pretrained_path = '/eagle/MDClimSim/tungnd/ClimaX/exps/global_forecast_climax/dinov2_vitl14_iterative_predict_diff/checkpoints/epoch_098.ckpt'


    log_dir = os.path.join(save_dir,'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    num_logs = len(os.listdir(log_dir))
    logging.basicConfig(filename=os.path.join(log_dir,'{}_{}.log'.format(save_dir_name,num_logs)),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info('Starting with analysis file : {}'.format(background_file_np))

    from varsClimaX import varsClimax
    vars_climax = varsClimax().vars_climax
    DEF_VARIABLES=varsClimax().DEF_VARIABLES

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'pressure']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0, 100.]
    obs_perc_err = [False, False, False, False, False, False]
    obs_err = ObsError(vars_climax, var_types, var_obs_err, obs_perc_err, stds)

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
    obs_dataset = ObsDatasetCum(filepath, start_date, end_date, vars_climax, 
                                obs_freq=obs_freq, da_window=da_window, 
                                obs_start_idx=0, obs_steps=obs_steps,
                                logger=logger)
    obs_loader = DataLoader(obs_dataset, batch_size=1, num_workers=0)
    
    #nn_model = ClimaX(vars_climax,
    #                  img_size=[128,256],
    #                  patch_size=4,
    #                  )
    nn_model = ClimaX(
        default_vars=DEF_VARIABLES, 
        img_size=[128, 256], 
        patch_size=4, 
        backbone='dinov2_vitl14', 
        vision_pretrained=False, 
        decoder_depth=2, 
        parallel_patch_embed=True)
    #print('nn_model.state_dict :',nn_model.state_dict())
    net_state_dict = nn_model.state_dict()
    print("Loading pre-trained checkpoint from: %s" % pretrained_path)
    checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))
    print('checkpoint.keys() :',checkpoint.keys())
    checkpoint_model = checkpoint["state_dict"]
    if "net.token_embeds.proj_weights" not in checkpoint_model.keys():
        raise ValueError(
            "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
        )
    # checkpoint_keys = list(checkpoint_model.keys())
    for k in list(checkpoint_model.keys()):
        if "channel" in k:
            checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
            del checkpoint_model[k]
    for k in list(checkpoint_model.keys()):
        if 'net.' in k:
            print(f"changing key {k} to {k.replace('net.','')} from pretrained checkpoint")
            #print(" - Shape of model", checkpoint_model[k].shape)
            checkpoint_model[k.replace('net.','')] = checkpoint_model[k]
            del checkpoint_model[k]
    for k in list(checkpoint_model.keys()):
        if k not in net_state_dict.keys() or checkpoint_model[k].shape != net_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            print(" - Shape of model", checkpoint_model[k].shape)
            print(" - Shape of entry", net_state_dict[k].shape)
            del checkpoint_model[k]
    nn_model.load_state_dict(checkpoint_model, strict=False)

    nn_model.to(device)
    nn_model.eval()

    climaX_Wrapper = ClimaXWrapper(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_climax,
        net=nn_model,
        device=device,
    )

    pytorch_total_params = sum(p.numel() for p in climaX_Wrapper.net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in climaX_Wrapper.net.parameters() if p.requires_grad)
    print('Total model parameters : {}'.format(pytorch_total_params))
    print('Trainable model parameters : {}'.format(pytorch_trainable_params))
    logger.info('Total model parameters : {}'.format(pytorch_total_params))
    logger.info('Trainable model parameters : {}'.format(pytorch_trainable_params))

    print('background_file_np :',background_file_np)
    logger.info('background_file_np : {}'.format(background_file_np))
    background_f = np.load(background_file_np, 'r')
    background = torch.from_numpy(background_f.copy())
    print('background_f.shape :',background_f.shape)
    print('background.shape :',background.shape)
    print('mean/min/max background temperature :',np.mean(background_f[0][0]),np.min(background_f[0][0]),np.max(background_f[0][0]))

    ########################################################################################################
    ########################################################################################################
    def run_forecast(x,
                    num_model_steps,
                    climaX_Wrapper=climaX_Wrapper,
                    vars_climax=vars_climax,
                    device=device
                    ):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        norm_preds,_,_ = climaX_Wrapper.forward_multi_step(
            x.to(device),
            vars_climax,
            num_model_steps)
        norm_preds = norm_preds[-1]
        return norm_preds

    def run_forecasts(x,
                    forecast_steps=2,
                    save_dir=save_dir,
                    noise_level=0.00,
                    climaX_Wrapper=climaX_Wrapper,
                    vars_climax=vars_climax,
                    device=device,
                    ):
        hf = h5py.File(os.path.join(save_dir, 'forecast_noise{}.h5'.format(noise_level)),'w')
        # forecasts (#forecasts,vars,lat,lon)
        forecasts = np.zeros((forecast_steps, x.shape[-3], x.shape[-2], x.shape[-1]))
        temp = torch.clone(x).detach()
        for i in range(forecast_steps):
            print('\tRunning forecast {}/{}'.format(i+1,forecast_steps))
            temp = run_forecast(temp,1,climaX_Wrapper,vars_climax,device)
            hf.create_dataset(str(i), data=temp.detach().cpu().numpy()[0])
        return
    ########################################################################################################
    ########################################################################################################

    #noise_levels = [0, 0.01, 0.1, 1, 10, 100, 1000]
    noise_levels = [0]

    all_obs, H_idxs, H_obs, shapes, obs_latlon = next(iter(obs_loader))
    all_obs = all_obs.detach().cpu().numpy()
    print('all_obs.shape :',all_obs.shape)
    H_idxs = H_idxs.detach().cpu().numpy()
    print('H_idxs.shape :',H_idxs.shape)
    print('H_idxs[:10] :',H_idxs[0,0,0,:10])
    H_idxs = H_idxs[:,:,:,::4]
    print('H_idxs.shape :',H_idxs.shape)
    print('H_idxs[:5] :',H_idxs[0,0,0,:5])

    diffs = np.zeros_like(background_f) # (1,1,82,128,256)
    H_idxs_unraveled_r, H_idxs_unraveled_c = np.unravel_index(H_idxs,(128,256)) # These hold the r,c of observation points
    print('H_idxs_unraveled_r :',H_idxs_unraveled_r.shape)
    print('min/max H_idxs_unraved_r/c',np.min(H_idxs_unraveled_r),np.max(H_idxs_unraveled_r),np.min(H_idxs_unraveled_c),np.max(H_idxs_unraveled_c))
    
    for v in range(len(vars_climax)):
        #print('all_obs[0,0,v,:] :',all_obs[0,0,v,:])
        #print('all_obs[0,0,v,:].shape :',all_obs[0,0,v,:].shape)
        print('min/max all_obs[0,0,{},:] : {}/{}'.format(v,np.min(all_obs[0,0,v,:]),np.max(all_obs[0,0,v,:])))
        for idx, (r,c) in enumerate(zip(H_idxs_unraveled_r[0,0,v],H_idxs_unraveled_c[0,0,v])):
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
    print('min/max diffs :',np.min(diffs),np.max(diffs))
    np.save(os.path.join(save_dir,'diffs'),diffs)
    diffs = torch.from_numpy(diffs)

    with torch.inference_mode():
        for noise_level in noise_levels:
            print('noise_level :',noise_level)
            background_perturbed = background + noise_level*diffs
            print('torch.mean/min/max background_perturbed :',torch.mean(background_perturbed),torch.min(background_perturbed),torch.max(background_perturbed))
            print('torch.mean/min/max temperature_perturbed :',torch.mean(background_perturbed[0][0]),torch.min(background_perturbed[0][0]),torch.max(background_perturbed[0][0]))
            run_forecasts(background_perturbed,
                          forecast_steps=40,
                          save_dir=save_dir,
                          noise_level=noise_level,
                          climaX_Wrapper=climaX_Wrapper,
                          vars_climax=vars_climax,
                          device=device,
                          )