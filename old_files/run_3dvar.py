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
from ml4dvar.climaX.climax_utils import ClimaXWrapper

from arch_swin import ClimaXSwin
from arch import ClimaX
#from ../Climax-v2/src/climax/arch_swin import ClimaXSwin

import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)
import logging

#NUM_MODEL_STEPS=2 #Current climax works at 6 hours so with assimilation wind of 12 hours we need to call the model twice

if __name__ == '__main__':

    start_date = datetime(2014, 1, 1, hour=0)
    end_date = datetime(2015, 12, 31, hour=12)
    da_window = 12
    model_step = 6
    obs_freq = 3
    da_type = 'var3d'
    #da_type = 'var4d'
    #save_dir = '/eagle/MDClimSim/mjp5595/data/var4d/'.format(da_type)
    #save_dir = '/eagle/MDClimSim/mjp5595/data/{}_cumObs/'.format(da_type)
    save_dir = '/eagle/MDClimSim/mjp5595/data/{}_cumObs3/'.format(da_type)

    filepath = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5" # Observations
    # TODO need Observations with all variables

    means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
    stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
    dv_param_file = '/eagle/MDClimSim/awikner/dv_params_128_256.hdf5'

    background_err_file = '/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/background_24hr_diff_sh_coeffs_var_climaxv2_standardized_128_uv.npy' #B (spherical harmonics)
    background_err_hf_file = '/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/background_24hr_diff_hf_var_climaxv2_standardized_128_uv.npy' #B (grid space (HF))

    b_inflation=1
    if da_type == 'var4d':
        b_inflation=1

    ####################################################################################################################################
    # Get start_idx for observations/analysis/background to start from
    ####################################################################################################################################
    background_file_np = '/eagle/MDClimSim/mjp5595/ml4dvar/background_starter.npy' # This is just to initialize the model background
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
    logging.basicConfig(filename=os.path.join(log_dir,'{}_{}.log'.format(da_type,start_idx+1)),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info('Starting with analysis file : {}'.format(background_file_np))

    from ml4dvar.climaX.vars_climaX import vars_climaX
    vars_climax = vars_climaX().vars_climax

    var_types = ['geopotential', 'temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind']
    var_obs_err = [100., 1.0, 1e-4, 1.0, 1.0]
    obs_perc_err = [False, False, False, False, False]
    obs_err = ObsError(vars_climax, var_types, var_obs_err, obs_perc_err, stds)
    print('obs_err :',obs_err.obs_err)

    # from src/dv.py
    dv_layer = DivergenceVorticity(vars_climax, means, stds, dv_param_file)

    background_err = torch.from_numpy(np.load(background_err_file)).float()
    background_err = background_err[torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]
    background_err_hf = torch.from_numpy(np.load(background_err_hf_file)).float()
    background_err_hf = background_err_hf[
        torch.concat((dv_layer.nowind_idxs, dv_layer.uwind_idxs, dv_layer.vwind_idxs))]

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
    obs_dataset = ObsDatasetCum(filepath, start_date, end_date, vars_climax, 
                                obs_freq=obs_freq, da_window=da_window, 
                                obs_start_idx=start_idx+1, obs_steps=obs_steps,
                                logger=logger)
    obs_loader = DataLoader(obs_dataset, batch_size=1, num_workers=0)
    
    nn_model = ClimaX(vars_climax,
                      img_size=[128,256],
                      patch_size=4,
                      )
    nn_model.to(device)
    nn_model.eval()

    climaX_Wrapper = ClimaXWrapper(
        root_dir='/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/',
        variables=vars_climax,
        net=nn_model,
        device=device,
    )

    print('background_file_np :',background_file_np)
    background_f = np.load(background_file_np, 'r')
    background = torch.from_numpy(background_f.copy())

    fourd_da = FourDVar(climaX_Wrapper, obs_loader,
                        background, background_err, background_err_hf,
                        obs_err, dv_layer, 
                        model_step=model_step,
                        da_window=da_window,
                        obs_freq=obs_freq,
                        da_type=da_type,
                        vars=vars_climax,
                        b_inflation=b_inflation,
                        savedir=save_dir,
                        device=device,
                        save_idx=start_idx+1,
                        logger=logger,
                        )
    fourd_da.fourDvar(forecast=True)