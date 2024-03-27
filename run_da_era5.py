import os, sys
from torch.utils.data import IterableDataset, DataLoader
import torch
from datetime import datetime, timedelta
import numpy as np
from src.dv import *
from src.obs_cummulative import ObsDatasetCum, ObsError 
from src.var_4d_reformatted import FourDVar

from stormer.models.hub.vit_adaln import ViTAdaLN
from stormer.stormer_utils import StormerWrapper
from stormer.stormer_utils_pangu import StormerWrapperPangu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_set = False
gpu2use = 0
if len(sys.argv) > 3:
    gpu2use = sys.argv[3]
    device = torch.device("cuda:{}".format(gpu2use) if torch.cuda.is_available() else "cpu")
    device_set = True
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

    exp_dir = '/eagle/MDClimSim/mjp5595/data/stormer/{}'.format(save_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    save_dir = '/eagle/MDClimSim/mjp5595/data/stormer/{}/data/'.format(save_dir_name)
    save_dir = os.path.join(exp_dir,'data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    obs_filepath = "/eagle/MDClimSim/mjp5595/ml4dvar/obs/igra_141520_stormer_obs_standardized_360_3.hdf5"

    means_file = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz'
    stds_file = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz'
    means = np.load(means_file)
    stds = np.load(stds_file)
    dv_param_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'

    b_inflation = 1 
    if da_type == 'var4d':
        b_inflation = 1

    background_err_file_dict = {}
    background_err_hf_file_dict = {}
    background_err_file_dict[0] = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/sh_12hr_stormer_vs_era5_00.npy'
    background_err_file_dict[12] = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/sh_12hr_stormer_vs_era5_12.npy'
    background_err_hf_file_dict[0] = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/hf_12hr_stormer_vs_era5_00.npy'
    background_err_hf_file_dict[12] = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/hf_12hr_stormer_vs_era5_12.npy'
    if device_set:
        if int(gpu2use) == 0:
            b_inflation = 10
        if int(gpu2use) == 1:
            b_inflation = 100
        if int(gpu2use) == 2:
            b_inflation = 1000
        if int(gpu2use) == 3:
            b_inflation = 10000

    ckpt_pth = '/eagle/MDClimSim/tungnd/stormer/models/6_12_24_climax_large_2_True_delta_8/checkpoints/epoch_015.ckpt'

    ####################################################################################################################################
    # Get start_idx for observations/analysis/background to start from
    ####################################################################################################################################
    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/stormer/data/background_init_stormer_norm_hr12.npy' # Init with 'random' era5 weather state from 1990
    backgrounds = os.listdir(save_dir)
    start_idx = 0
    if len(backgrounds) > 1:
        for background_file in backgrounds:
            if 'background' in background_file:
                background_num = int(background_file.split('_')[1])
                if background_num > start_idx:
                    start_idx = background_num
                    background_file_np = os.path.join(save_dir,background_file)
    print('Starting with background file : {}'.format(background_file_np))
    ####################################################################################################################################

    log_dir = os.path.join(exp_dir,'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    num_logs = len(os.listdir(log_dir))
    logging.basicConfig(filename=os.path.join(log_dir,'{}_{}.log'.format(save_dir_name,num_logs)),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger.info('')
    logger.info('')
    logger.info('da_type : {}'.format(da_type))
    logger.info('save_dir_name : {}'.format(save_dir_name))
    logger.info('')
    logger.info('b_inflation : {}'.format(b_inflation))
    logger.info('Using checkpoint pth : {}'.format(ckpt_pth))
    logger.info('obs_filepath : {}'.format(obs_filepath))
    logger.info('means_file : {}'.format(means_file))
    logger.info('stds_file : {}'.format(stds_file))
    logger.info('dv_param file : {}'.format(dv_param_file))
    logger.info('')
    logger.info('Using background_err_file_dict : {}'.format(background_err_file_dict))
    logger.info('Using background_err_hf_file_dict : {}'.format(background_err_hf_file_dict))
    logger.info('Starting with background file : {}'.format(background_file_np))
    logger.info('')

    from stormer.varsStormer import varsStormer
    vars_stormer = varsStormer().vars_stormer

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'pressure']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0, 100.]
    obs_perc_err = [False, False, False, False, False, False]
    obs_err = ObsError(vars_stormer, var_types, var_obs_err, obs_perc_err, stds, device)
    print('obs_err :',obs_err.obs_err)
    if logger:
        logger.info('obs_err : {}'.format(obs_err.obs_err))

    # from src/dv.py
    dv_layer = DivergenceVorticity(vars_stormer, means, stds, dv_param_file, device)

    def make_bgerr_dict(bg_err_file_dict):
        bg_err_dict = {}
        for hr_key in bg_err_file_dict.keys():
            bg_err_file = bg_err_file_dict[hr_key]
            be = np.load(bg_err_file)
            background_err = torch.from_numpy(be).float().to(device)
            background_err = background_err[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]
            bg_err_dict[int(hr_key)] = background_err
        return bg_err_dict

    background_err_dict = make_bgerr_dict(background_err_file_dict)
    background_err_hf_dict = make_bgerr_dict(background_err_hf_file_dict)

    ## Set B to identity matrix
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
    use_only_recent_obs = False
    if da_type == 'var3d':
        use_only_recent_obs = True
    obs_steps = 1
    if da_type == 'var4d':
        obs_steps = da_window // model_step
    obs_dataset = ObsDatasetCum(obs_filepath, start_date, end_date, vars_stormer, 
                                obs_freq=obs_freq, da_window=da_window, 
                                obs_start_idx=start_idx, obs_steps=obs_steps,
                                only_recent_obs=use_only_recent_obs, logger=logger,
                                device=device)
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
    #if '.npy' in background_file_np:
    background_f = np.load(background_file_np, 'r')
    #elif '.h5' in background_file_np:
    #    background_f = h5py.File(background_file_np, 'r')[str(forecast_hrs)][:]
    #    background_f = np.expand_dims(background_f,axis=0)
    if 'rand' in save_dir:
        #background_f = np.zeros_like(background_f)
        print('using random background')
        background_f = np.random.randn(*background_f.shape)
    background = torch.from_numpy(background_f.copy())

    fourd_da = FourDVar(stormer_wrapper, obs_loader,
                        background, background_err_dict, background_err_hf_dict,
                        obs_err, dv_layer, 
                        model_step=model_step,
                        da_window=da_window,
                        obs_freq=obs_freq,
                        da_type=da_type,
                        vars=vars_stormer,
                        b_inflation=b_inflation,
                        max_iter=700,
                        savedir=save_dir,
                        device=device,
                        save_idx=start_idx,
                        logger=logger,
                        )
    fourd_da.cycleDataAssimilation(forecast=True)