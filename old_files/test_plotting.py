from src.plotting import *
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

means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
#obs_file = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"
obs_file = "/eagle/MDClimSim/troyarcomano/ml4dvar_climax_v2/igra_141520_stormer_obs_standardized.hdf5"
obs_file2 = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')

start_date = datetime(2014, 1, 1, hour=0)

analysis = AnalysisData(start_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)
print(analysis.analysis.shape)
era5 = ERA5Data(start_date, analysis.end_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)
print(era5.data.shape)

obs = ObsData(start_date, analysis.end_date, 12, vars, obs_file, means = means, stds = stds, lat = lat, lon = lon)

obs2 = ObsData(start_date, analysis.end_date, 12, vars, obs_file2, means = means, stds = stds, lat = lat, lon = lon)

forecasts = ForecastData(start_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)

#_ = plot_analysis_innovation(era5, analysis, obs, vars_units, var_idxs = [3], itr_idxs = np.arange(30), save = True, show = False, save_dir='/eagle/MDClimSim/troyarcomano/ml4dvar/plots/',return_error=False)

era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background = plot_observations(era5, analysis, obs, obs2, vars_units, var_idxs = np.arange(len(vars)), itr_idxs = np.arange(30), save = True, show = False, save_dir='/eagle/MDClimSim/troyarcomano/ml4dvar/plots/',return_error=True)
#era5_minus_analysis, era5_obs, era5_obs_error, analysis_obs, analysis_obs_error, era5_minus_background = plot_analysis(era5, analysis, obs, vars_units, var_idxs = np.arange(len(vars)), itr_idxs = np.arange(30), save = False, show = False, save_dir='/eagle/MDClimSim/troyarcomano/ml4dvar/plots/',return_error=True)

#plot_analysis_global_rmse(era5_minus_analysis, era5_minus_background, vars, vars_units,  var_id=3, lat_weighted=True, lats=lat)
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
