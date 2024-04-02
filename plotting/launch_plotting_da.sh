#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate climaX
cd /eagle/MDClimSim/mjp5595/ml4dvar/plotting
python plotting_da.py var3d_era5B_era5Obs_Jan15_gpu0
python plotting_da.py var3d_era5B_era5Obs_Jan15_gpu1
python plotting_da.py var3d_NMC12hrB_era5Obs_Jan15
python plotting_da.py var3d_NMC24hrB_era5Obs_Jan15