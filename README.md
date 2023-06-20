# climaX_4dvar
Repository for 4D Var data assimilation with ClimaX

# Installation
Install the environment:

`conda create -f environment.yml`

You will probably need to install  `torch-harmonics` from source. After activating the `climaX_4dvar` environment, run:

`cd torch-harmonics`

`pip install -e .`

# Downloading and Processing Observation Data
This code assimilates radiosonde observations from the [Integrated global radiosonde Archive (IGRA)](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive).
This data may be downloaded from this [link](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/archive/IGRA_v2.2_data-por_s19050404_e20230619_c20230619.tar).
The data is contained in a .tar file, which contains a number of other .zip files with the radiosonde data from each station.
We first process this data into an .hdf5 file by running `process-obs/expand_data.py` to unzip the data files, then 
`process_data/load_obs.py` to extract the data for a prescribed set of years into a set of .hdf5 files, then run
`process_data/combine_hdf5_files.py` to combine all of these into a single .hdf5 file. One can delete the intermediate
.hdf5 files after this has finished.

We then run `process-data/add_sh.py` to extract only the data at the pressure levels our model predicts, convert the data
from the observed format to our model format, and compute specific humidity from dewpoint depression, pressure,
and temperature. Finally, we run `process-data/compute_H_obs.py` to compute the observation operator model indices and
interpolation weights and combine this into a final .hdf5 data file. If you don't want any of the intermediate .hdf5 files,
they may be deleted at this point.
