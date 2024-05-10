import os, sys
from torch.utils.data import IterableDataset, DataLoader
import torch
from datetime import datetime, timedelta
import numpy as np
from src.dv import *
from src.obs_cummulative import ObsDatasetCum, ObsError 
from src.var_4d_reformatted import FourDVar
import shutil

from stormer.models.hub.vit_adaln import ViTAdaLN
from stormer.stormer_utils import StormerWrapper
from stormer.stormer_utils_pangu import StormerWrapperPangu
import yaml

print('sys.argv :',sys.argv)
print('torch.cuda.is_available() :',torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_set = False
gpu2use = 0
if len(sys.argv) > 2:
    gpu2use = int(sys.argv[2])
    device = torch.device("cuda:{}".format(gpu2use) if torch.cuda.is_available() else "cpu")
    device_set = True
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)
import logging

if __name__ == '__main__':

    config_pth = str(sys.argv[1])
    with open(config_pth,'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    da_type = config['da']['da_type']
    save_dir_name = config['da']['save_dir_name']

    start_date = datetime(config['obs']['start']['year'],
                          config['obs']['start']['month'],
                          config['obs']['start']['day'],
                          hour=config['obs']['start']['hour'])
    end_date = datetime(config['obs']['end']['year'],
                          config['obs']['end']['month'],
                          config['obs']['end']['day'],
                          hour=config['obs']['end']['hour'])
    da_window = config['da']['da_window'] 
    model_step = config['model']['model_step']
    #obs_freq = config['obs']['obs_freq']

    da_root_dir = config['da']['da_root_dir']
    obs_filepath = config['obs']['obs_filepath']

    means_file = config['data']['means_file']
    stds_file = config['data']['stds_file']
    means = np.load(means_file)
    stds = np.load(stds_file)
    dv_param_file = config['data']['dv_param_file']

    b_inflation = config['da']['b_inflation'] 
    b_hf_inflation = config['da']['b_hf_inflation'] 
    
    replace_uvwinds = False
    try:
        replace_uvwinds = config['da']['replace_uvwinds']
    except:
        pass
    era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'

    background_err_file_dict = {}
    background_err_hf_file_dict = {}
    for hr_key in config['error_covariance']['background_err_file'].keys():
        background_err_file_dict[int(hr_key)] = config['error_covariance']['background_err_file'][hr_key]
        background_err_hf_file_dict[int(hr_key)] = config['error_covariance']['background_err_hf_file'][hr_key]

    ckpt_pth = config['model']['ckpt_pth']
    max_iter = config['da']['max_iter']

    model_name = config['model']['model_name']

    exp_dir = os.path.join(da_root_dir,'data',model_name,'{}'.format(save_dir_name))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    shutil.copyfile(config_pth,os.path.join(exp_dir,os.path.basename(config_pth)))

    save_dir = os.path.join(da_root_dir,'data',model_name,'{}'.format(save_dir_name),'data')
    save_dir = os.path.join(exp_dir,'data')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ####################################################################################################################################
    # Get start_idx for observations/analysis/background to start from
    ####################################################################################################################################
    background_file_np = config['data']['background_file_np']
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

    print('background_file_np :',background_file_np)
    background_f = np.load(background_file_np, 'r')
    if 'rand' in save_dir:
        print('using random background')
        background_f = np.random.randn(*background_f.shape)
    background = torch.from_numpy(background_f.copy())

    from stormer.varsStormer import varsStormer
    vars_stormer = varsStormer().vars_stormer
    uwind_idxs = varsStormer().uwind_idxs
    vwind_idxs = varsStormer().vwind_idxs
    nowind_idxs = varsStormer().nowind_idxs
    # from src/dv.py

    wind_type = None
    try:
        wind_type = config['da']['wind_type']
        if wind_type == 'vector':
            wind_layer = UVWwind(background_f, vars_stormer, stds, device)
        elif wind_type == 'scalar':
            wind_layer = DivergenceVorticity(background_f, vars_stormer, means, stds, dv_param_file, device)
        elif wind_type == 'uv_scalar':
            wind_layer = UVScalar(background_f, vars_stormer, stds, device)
            wind_type = 'scalar'
    except:
        pass
    if wind_type is None:
        wind_layer = DivergenceVorticity(vars_stormer, means, stds, dv_param_file, device)
        wind_type = 'scalar'

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
    logger.info('Using device : {}'.format(device))
    logger.info('da_type : {}'.format(da_type))
    logger.info('save_dir_name : {}'.format(save_dir_name))
    logger.info('')
    logger.info('b_inflation : {}'.format(b_inflation))
    logger.info('Using checkpoint pth : {}'.format(ckpt_pth))
    logger.info('obs_filepath : {}'.format(obs_filepath))
    logger.info('means_file : {}'.format(means_file))
    logger.info('stds_file : {}'.format(stds_file))
    logger.info('max_iter : {}'.format(max_iter))
    logger.info('')
    logger.info('using wind type : {}'.format(wind_type))
    logger.info('dv_param file : {}'.format(dv_param_file))
    logger.info('replace uvwinds : {}'.format(replace_uvwinds))
    logger.info('')
    logger.info('Using background_err_file_dict : {}'.format(background_err_file_dict))
    logger.info('Using background_err_hf_file_dict : {}'.format(background_err_hf_file_dict))
    logger.info('Starting with background file : {}'.format(background_file_np))
    logger.info('')

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'pressure']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0, 100.]
    obs_perc_err = [False, False, False, False, False, False]
    obs_err = ObsError(vars_stormer, var_types, var_obs_err, obs_perc_err, stds, device)
    print('obs_err :',obs_err.obs_err)
    if logger:
        logger.info('obs_err : {}'.format(obs_err.obs_err))


    def make_bgerr_dict(bg_err_file_dict):
        bg_err_dict = {}
        for hr_key in bg_err_file_dict.keys():
            bg_err_file = bg_err_file_dict[hr_key]
            be = np.load(bg_err_file)
            background_err = torch.from_numpy(be).float().to(device)
            background_err = background_err[torch.concat((torch.Tensor(nowind_idxs).long().to(device),
                                                         torch.Tensor(uwind_idxs).long().to(device),
                                                         torch.Tensor(vwind_idxs).long().to(device)
                                                         )
                                                         )]
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
    use_only_recent_obs = config['obs']['use_only_recent_obs']
    obs_steps = config['obs']['obs_steps']
    obs_dataset = ObsDatasetCum(obs_filepath, start_date, end_date, vars_stormer, 
                                #obs_freq=obs_freq, da_window=da_window, 
                                da_window=da_window, 
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

    # TODO copy to own data dir
    stormer_wrapper = StormerWrapperPangu(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        #root_dir=os.path.join(da_root_dir,'data','wb2','1.40625deg_from_full_res_1_step_6hr_h5df')
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

    fourd_da = FourDVar(stormer_wrapper, obs_loader,
                        background, background_err_dict, background_err_hf_dict,
                        obs_err, wind_layer, 
                        wind_type=wind_type,
                        model_step=model_step,
                        da_window=da_window,
                        #obs_freq=obs_freq,
                        da_type=da_type,
                        vars=vars_stormer,
                        b_inflation=b_inflation,
                        b_hf_inflation=b_hf_inflation,
                        max_iter=max_iter,
                        savedir=save_dir,
                        device=device,
                        save_idx=start_idx,
                        logger=logger,
                        )
    if replace_uvwinds:
        #fourd_da.cycleDataAssimilation(forecast=True,era5_dir=era5_dir,stds=stds)
        fourd_da.cycleDataAssimilation(forecast=False,era5_dir=era5_dir,stds=stds)
    else:
        fourd_da.cycleDataAssimilation(forecast=True)