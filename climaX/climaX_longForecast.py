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
#device = "cpu"
print('USING DEVICE :',device)

torch.autograd.set_detect_anomaly(True)
import logging


if __name__ == '__main__':

    save_dir_name = 'climaX_longForecast'

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

    from ml4dvar.climaX.vars_climaX import vars_climaX
    vars_climax = vars_climaX().vars_climax

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
                    climaX_Wrapper=climaX_Wrapper,
                    vars_climax=vars_climax,
                    device=device,
                    ):
        hf = h5py.File(os.path.join(save_dir, 'longForecast.h5'),'w')
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

    with torch.inference_mode():
        run_forecasts(background,
                        forecast_steps=1000,
                        save_dir=save_dir,
                        climaX_Wrapper=climaX_Wrapper,
                        vars_climax=vars_climax,
                        device=device,
                        )