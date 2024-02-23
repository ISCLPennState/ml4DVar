import os, sys
from torch.utils.data import IterableDataset, DataLoader
import torch
from datetime import datetime, timedelta
import numpy as np
from src.dv import *
from src.obs_cummulative import ObsDatasetCum, ObsError 
from src.var_4d_reformatted import FourDVar

from stormer.models.hub.vit_adaln import ViTAdaLN
from stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from stormer.stormer_utils import StormerWrapper
from stormer.stormer_utils_pangu import StormerWrapperPangu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)
import logging

if __name__ == '__main__':

    #da_type = ['var4d','var3d']
    da_type = str(sys.argv[1])
    save_dir_name = str(sys.argv[2])

    start_date = datetime(2014, 1, 1, hour=0)
    end_date = datetime(2015, 12, 31, hour=12)
    da_window = 12
    model_step = 6
    obs_freq = 3
    save_dir = '/eagle/MDClimSim/mjp5595/data/stormer/{}/'.format(save_dir_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/obs/igra_141520_stormer_obs_standardized_360_2.hdf5"

    means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
    dv_param_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'

    background_err_file = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/background_24hr_diff_sh_coeffs_var_stormer_standardized_128_uv.npy' #B (spherical harmonics)
    background_err_hf_file = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/background_24hr_diff_hf_var_stormer_standardized_128_uv.npy' #B (grid space (HF))

    ckpt_pth = '/eagle/MDClimSim/tungnd/stormer/models/6_12_24_climax_large_2_True_delta_8/checkpoints/epoch_015.ckpt'

    b_inflation = 1
    if da_type == 'var4d':
        b_inflation = 1

    ####################################################################################################################################
    # Get start_idx for observations/analysis/background to start from
    ####################################################################################################################################
    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/background_init_stormer_norm.npy' # Init with 'random' era5 weather state from 1990
    analyses = os.listdir(save_dir)
    analysis_files = analyses
    start_idx = -1
    if len(analyses) > 1:
        for analysis_file in analysis_files:
            if 'analysis' in analysis_file:
                analysis_num = int(analysis_file.split('_')[1])
                if analysis_num > start_idx:
                    start_idx = analysis_num
                    background_file_np = os.path.join(save_dir,analysis_file)
    print('Starting with analysis file :',background_file_np)
    ####################################################################################################################################

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

    from stormer.varsStormer import varsStormer
    vars_stormer = varsStormer().vars_stormer

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'pressure']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0, 100.]
    obs_perc_err = [False, False, False, False, False, False]
    obs_err = ObsError(vars_stormer, var_types, var_obs_err, obs_perc_err, stds)
    print('obs_err :',obs_err.obs_err)

    # from src/dv.py
    dv_layer = DivergenceVorticity(vars_stormer, means, stds, dv_param_file)

    background_err = torch.from_numpy(np.load(background_err_file)).float().to(device)
    background_err = background_err[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]
    background_err_hf = torch.from_numpy(np.load(background_err_hf_file)).float().to(device)
    background_err_hf = background_err_hf[
        torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]

    # Set B to identity matrix
    #print('background_err.shape (0):',background_err.shape)
    #a,b = background_err.shape
    #background_err = torch.eye(a,b)
    #print('background_err.shape (1):',background_err.shape)
    #background_err = background_err + 1e-6

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
    if da_type == 'var4d':
        obs_steps = da_window // model_step
    obs_dataset = ObsDatasetCum(filepath, start_date, end_date, vars_stormer, 
                                obs_freq=obs_freq, da_window=da_window, 
                                obs_start_idx=start_idx+1, obs_steps=obs_steps,
                                logger=logger)
    obs_loader = DataLoader(obs_dataset, batch_size=1, num_workers=0)

    ###################################################################################################################
    ###################################################################################################################
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
    #stormer_wrapper = StormerWrapper(
    #    root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
    #    variables=vars_stormer,
    #    net=net,
    #    list_lead_time=[6],
    #    ckpt=ckpt_pth,
    #    device=device,
    #)
    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_stormer,
        net=net,
        base_lead_time=6,
        possible_lead_times=[24,12,6],
        ckpt=ckpt_pth,
        device=device,
        logger=logger,
    )

    pytorch_total_params = sum(p.numel() for p in stormer_wrapper.net.parameters())
    pytorch_trainable_params = sum(p.numel() for p in stormer_wrapper.net.parameters() if p.requires_grad)
    print('Total model parameters : {}'.format(pytorch_total_params))
    print('Trainable model parameters : {}'.format(pytorch_trainable_params))
    logger.info('Total model parameters : {}'.format(pytorch_total_params))
    logger.info('Trainable model parameters : {}'.format(pytorch_trainable_params))

    print('background_file_np :',background_file_np)
    logger.info('background_file_np : {}'.format(background_file_np))
    background_f = np.load(background_file_np, 'r')
    if 'rand' in save_dir:
        #background_f = np.zeros_like(background_f)
        print('using random background')
        background_f = np.random.randn(*background_f.shape)
    background = torch.from_numpy(background_f.copy())

    fourd_da = FourDVar(stormer_wrapper, obs_loader,
                        background, background_err, background_err_hf,
                        obs_err, dv_layer, 
                        model_step=model_step,
                        da_window=da_window,
                        obs_freq=obs_freq,
                        da_type=da_type,
                        vars=vars_stormer,
                        b_inflation=b_inflation,
                        max_iter=200,
                        savedir=save_dir,
                        device=device,
                        save_idx=start_idx+1,
                        logger=logger,
                        )
    fourd_da.fourDvar(forecast=True)