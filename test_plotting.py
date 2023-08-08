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

means = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_mean.npz')
stds = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/normalize_std.npz')
obs_file = "/eagle/MDClimSim/awikner/irga_1415_test1_obs.hdf5"

lat = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lat.npy')
lon = np.load('/eagle/MDClimSim/troyarcomano/1.40625deg_npz_40shards/lon.npy')

start_date = datetime(2014, 1, 1, hour=0)

analysis = AnalysisData(start_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)
print(analysis.analysis.shape)
era5 = ERA5Data(start_date, analysis.end_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)
print(era5.data.shape)

obs = ObsData(start_date, analysis.end_date, 12, vars, obs_file, means = means, stds = stds, lat = lat, lon = lon)

forecasts = ForecastData(start_date, 12, vars, means = means, stds = stds, lat = lat, lon = lon)

plot_analysis(era5, analysis, obs, var_idxs = np.array([0]), save = False, show = True)
