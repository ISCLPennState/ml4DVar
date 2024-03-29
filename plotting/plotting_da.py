from plotting_360obs import *
import numpy as np
from datetime import *
import sys

sys.path.append("/eagle/MDClimSim/mjp5595/ml4dvar/")
from stormer.varsStormer import varsStormer
vars = varsStormer().vars_stormer
var_units = varsStormer().var_units

means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
obs_file = "/eagle/MDClimSim/mjp5595/ml4dvar/obs/igra_141520_stormer_obs_standardized_360_3.hdf5"

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')

obs_start_date = datetime(2014, 1, 1, hour=0)
analysis_start_date = datetime(2014, 1, 1, hour=12)

base_dir = '/eagle/MDClimSim/mjp5595/data/stormer/'
if len(sys.argv) > 1:
    exp_dir = os.path.join(base_dir,sys.argv[1])
    #exp_dir = os.path.join(base_dir,'var3d_BhrDiff_gpu1')
else:
    exp_dir = os.path.join(base_dir,'stormer3d')

save_dir = os.path.join(exp_dir,'data')
print('save_dir :',save_dir)
plot_dir = os.path.join(exp_dir,'plots')
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
da_window = 12
#max_steps_to_plot = 20
max_steps_to_plot = 100
#max_steps_to_plot = 10

era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'

analysis = AnalysisData(analysis_start_date,
                        time_step=da_window, 
                        vars=vars, 
                        dir=save_dir, 
                        means=means, 
                        stds=stds, 
                        lat=lat, 
                        lon=lon)
num_windows,_,_,_ = analysis.analysis.shape
print('analysis.shape :',analysis.analysis.shape)

era5 = ERA5Data(analysis_start_date,
                da_window=da_window,
                data_freq=6,
                num_windows=num_windows,
                vars=vars,
                dir=era5_dir,
                means=means,
                stds=stds,
                lat=lat,
                lon=lon)
print('era5.shape :',era5.data.shape)

obs = ObsData(obs_start_date,
              analysis.end_date,
              12,
              vars,
              obs_file,
              means = means,
              stds = stds,
              lat = lat,
              lon = lon)
print('obs.obs :',len(obs.obs))

#forecasts = ForecastData(analysis_start_date,
#                         12,
#                         vars,
#                         dir=save_dir,
#                         means = means,
#                         stds = stds,
#                         lat = lat,
#                         lon = lon)
#print('forecasts.forecasts.shape :',forecasts.forecasts.shape)

ana_dir = os.path.join(plot_dir,'analysis')
if not os.path.exists(ana_dir):
    os.makedirs(ana_dir)
plot_stuff = plot_analysis(era5,
                          analysis,
                          obs,
                          var_units,
                          #var_idxs = [0,3,11],
                          var_idxs = [0],
                          #var_idxs = None,
                          window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                          #window_idxs = [0],
                          save = True,
                          show = False,
                          save_dir = ana_dir,
                          return_error = True)
era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background = plot_stuff
print('Done with plot_analysis')

bg_vs_ana_dir = os.path.join(plot_dir,'bg_vs_ana')
if not os.path.exists(bg_vs_ana_dir):
    os.makedirs(bg_vs_ana_dir)
_ = plot_background_vs_analysis(era5,
                                analysis,
                                obs,
                                var_units,
                                #var_idxs = [0,3,11],
                                var_idxs = [0],
                                #var_idxs = None,
                                window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                                #window_idxs = [0],
                                save = True,
                                show = False,
                                save_dir = bg_vs_ana_dir,
                                return_error = True)
####################################################################################################################

innovation_dir = os.path.join(plot_dir,'innovation')
if not os.path.exists(innovation_dir):
    os.makedirs(innovation_dir)
_ = plot_analysis_innovation(era5,
                             analysis,
                             obs,
                             var_units,
                             #var_idxs = [0,3,11],
                             var_idxs = [0],
                             window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                             save = True,
                             show = False,
                             save_dir = innovation_dir,
                             return_error = False,
                             plot_obs = True)

_ = plot_analysis_innovation(era5,
                             analysis,
                             obs,
                             var_units,
                             #var_idxs = [0,3,11],
                             var_idxs = [0],
                             window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                             save = True,
                             show = False,
                             save_dir = innovation_dir,
                             return_error = False,
                             plot_obs = False)

mse_dir = os.path.join(plot_dir,'mse')
if not os.path.exists(mse_dir):
    os.makedirs(mse_dir)
plot_analysis_global_rmse(era5_minus_analysis,
                          era5_minus_background,
                          analysis,
                          vars,
                          var_units, 
                          #var_id = 3, 
                          var_idxs = [0,3,11],
                          window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                          #lat_weighted = True, 
                          lat_weighted = True, 
                          lats = lat,
                          save = True,
                          show = False,
                          save_dir = mse_dir,
                          return_error = False)