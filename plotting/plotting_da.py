from plotting_360obs import *
import numpy as np
from datetime import *

vars = ['2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'geopotential_500',
            'geopotential_700',
            'geopotential_850',
            'geopotential_925',
            'u_component_of_wind_250',
            'u_component_of_wind_500',
            'u_component_of_wind_700',
            'u_component_of_wind_850',
            'u_component_of_wind_925',
            'v_component_of_wind_250',
            'v_component_of_wind_500',
            'v_component_of_wind_700',
            'v_component_of_wind_850',
            'v_component_of_wind_925',
            'temperature_250',
            'temperature_500',
            'temperature_700',
            'temperature_850',
            'temperature_925',
            'specific_humidity_250',
            'specific_humidity_500',
            'specific_humidity_700',
            'specific_humidity_850',
            'specific_humidity_925']


vars2 = ["2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "mean_sea_level_pressure",
        "geopotential_50",
        "geopotential_100",
        "geopotential_150",
        "geopotential_200",
        "geopotential_250",#
        "geopotential_300",
        "geopotential_400",
        "geopotential_500",
        "geopotential_600",
        "geopotential_700",
        "geopotential_850",
        "geopotential_925",
        "geopotential_1000",
        "u_component_of_wind_50",
        "u_component_of_wind_100",
        "u_component_of_wind_150",
        "u_component_of_wind_200",
        "u_component_of_wind_250",
        "u_component_of_wind_300",
        "u_component_of_wind_400",
        "u_component_of_wind_500",
        "u_component_of_wind_600",
        "u_component_of_wind_700",
        "u_component_of_wind_850",
        "u_component_of_wind_925",
        "u_component_of_wind_1000",
        "v_component_of_wind_50",
        "v_component_of_wind_100",
        "v_component_of_wind_150",
        "v_component_of_wind_200",
        "v_component_of_wind_250",
        "v_component_of_wind_300",
        "v_component_of_wind_400",
        "v_component_of_wind_500",
        "v_component_of_wind_600",
        "v_component_of_wind_700",
        "v_component_of_wind_850",
        "v_component_of_wind_925",
        "v_component_of_wind_1000",
        "vertical_velocity_50",#   
        "vertical_velocity_100",#  
        "vertical_velocity_150",#
        "vertical_velocity_200",#
        "vertical_velocity_250",#
        "vertical_velocity_300",#
        "vertical_velocity_400",#
        "vertical_velocity_500",#
        "vertical_velocity_600",#
        "vertical_velocity_700",#
        "vertical_velocity_850",#
        "vertical_velocity_925",#
        "vertical_velocity_1000",# unmeasurable
        "temperature_50",
        "temperature_100",
        "temperature_150",
        "temperature_200",
        "temperature_250",
        "temperature_300",
        "temperature_400",
        "temperature_500",
        "temperature_600",
        "temperature_700",
        "temperature_850",
        "temperature_925",
        "temperature_1000",
        "specific_humidity_50",
        "specific_humidity_100",
        "specific_humidity_150",
        "specific_humidity_200",
        "specific_humidity_250",
        "specific_humidity_300",
        "specific_humidity_400",
        "specific_humidity_500",
        "specific_humidity_600",
        "specific_humidity_700",
        "specific_humidity_850",
        "specific_humidity_925",
        "specific_humidity_1000",
        ]

vars_units = ['K',
            'm/s',
            'm/s',
            'm^2/s^2',
            'm^2/s^2',
            'm^2/s^2',
            'm^2/s^2',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'm/s',
            'K',
            'K',
            'K',
            'K',
            'K',
            'kg/kg',
            'kg/kg',
            'kg/kg',
            'kg/kg',
            'kg/kg']

#means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
#stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
#obs_file = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"

means = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/normalize_std.npz')
#obs_file = "/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/igra_141520_stormer_obs_standardized.hdf5"
#obs_file = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized.hdf5"
obs_file = "/eagle/MDClimSim/mjp5595/ml4dvar/igra_141520_stormer_obs_standardized_360_2.hdf5"

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')
print('lon min/max (0):',np.min(lon),np.max(lon))
#lon = lon - 180
print('lon min/max (1):',np.min(lon),np.max(lon))

obs_start_date = datetime(2014, 1, 1, hour=0)
analysis_start_date = datetime(2014, 1, 1, hour=12)

#save_dir = '/eagle/MDClimSim/mjp5595/data/var3d_normForecast/'
#save_dir = '/eagle/MDClimSim/mjp5595/data/var4d_mattObs/'
#save_dir = '/eagle/MDClimSim/mjp5595/data/test_3D_ObsLoader_ds360_2/'
#save_dir = '/eagle/MDClimSim/mjp5595/data/var4d_defVars/'
save_dir = '/eagle/MDClimSim/mjp5595/data/var4d_eyeB/'
da_window = 12
max_steps_to_plot = 30

#era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_1_step_6hr_h5df/train/'
era5_dir = '/eagle/MDClimSim/tungnd/data/wb2/1.40625deg_from_full_res_1_step_6hr_h5df/train/'
#era5_dir = '/eagle/MDClimSim/tungnd/data/datasets/climate/wb2/1.40625deg_6hr_h5df/train/'

analysis = AnalysisData(analysis_start_date,
                        time_step=da_window, 
                        vars=vars2, 
                        dir=save_dir, 
                        means=means, 
                        stds=stds, 
                        lat=lat, 
                        lon=lon)
num_windows,_,_,_ = analysis.analysis.shape
print('analysis.shape :',analysis.analysis.shape)
#era5 = ERA5Data(start_date,
#                analysis.end_date,
#                time_step=12,
#                vars=vars2,
#                dir=era5_dir,
#                shards=min(40,times),
#                means=means,
#                stds=stds,
#                lat=lat,
#                lon=lon)
era5 = ERA5Data(analysis_start_date,
                da_window=da_window,
                data_freq=6,
                num_windows=num_windows,
                vars=vars2,
                dir=era5_dir,
                means=means,
                stds=stds,
                lat=lat,
                lon=lon)
print('era5.shape :',era5.data.shape)

obs = ObsData(obs_start_date,
              analysis.end_date,
              12,
              vars2,
              obs_file,
              means = means,
              stds = stds,
              lat = lat,
              lon = lon)
print('obs.obs :',len(obs.obs))

forecasts = ForecastData(analysis_start_date,
                         12,
                         vars2,
                         dir=save_dir,
                         means = means,
                         stds = stds,
                         lat = lat,
                         lon = lon)
print('forecasts.forecasts.shape :',forecasts.forecasts.shape)

#era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background = plot_analysis(era5, analysis, obs, vars_units, var_idxs = np.arange(len(vars)), itr_idxs = np.arange(30), save = False, show = False, save_dir='/eagle/MDClimSim/troyarcomano/ml4dvar/plots/',return_error=True)
plot_stuff = plot_analysis(era5,
                          analysis,
                          obs,
                          vars_units,
                          #var_idxs = np.arange(len(vars2)),
                          var_idxs = [0],
                          window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                          save = True,
                          show = False,
                          save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/plots/',
                          return_error = True)
era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background = plot_stuff
print('Done with plot_analysis')

_ = plot_analysis_innovation(era5,
                             analysis,
                             obs,
                             vars_units,
                             var_idxs = [0],
                             window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                             save = True,
                             show = False,
                             save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/plots/',
                             return_error = False)

plot_analysis_global_rmse(era5_minus_analysis,
                          era5_minus_background,
                          analysis,
                          vars2,
                          vars_units, 
                          #var_id = 3, 
                          var_idxs = [0],
                          window_idxs = np.arange(min(max_steps_to_plot,num_windows)),
                          lat_weighted = True, 
                          lats = lat,
                          save = True,
                          show = False,
                          save_dir = '/eagle/MDClimSim/mjp5595/ml4dvar/plots/',
                          return_error = False)

#_ = plot_analysis_innovation(era5, analysis, obs, vars_units, var_idxs = [3], itr_idxs = np.arange(30), save = True, show = False, save_dir='/eagle/MDClimSim/troyarcomano/ml4dvar/plots/',return_error=False)
'''
print(np.shape(analysis_obs_error))
print(np.shape(analysis_obs_error[:,0]))
print(np.shape(analysis_obs_error[0,0]))
rmse = np.zeros((np.shape(analysis_obs_error)[0]))
rmse_era = np.zeros((np.shape(analysis_obs_error)[0]))
for i in range(np.shape(analysis_obs_error)[0]):
    rmse[i] = np.sqrt(np.mean(analysis_obs_error[i,0]**2))
    rmse_era[i] = np.sqrt(np.mean(era5_obs_error[i,0]**2))
#print(analysis_obs_error[:,0])
plt.plot(rmse,label='Analysis')
plt.plot(rmse_era,label='ERA5')
plt.legend()
plt.show()
'''
