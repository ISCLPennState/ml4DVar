#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=6:00:00
#PBS -q preemptable
#PBS -A NeuralDE
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

module use /soft/modulefiles
ml conda
conda activate climaX
cd /eagle/MDClimSim/mjp5595/ml4dvar/plotting
python plotting_da.py var3d_4cellGrid_windVec
python plotting_da.py var3d_denseGrid_B1
python plotting_da.py var3d_sparseERA5_vecWind
python plotting_da.py var3d_4cellGrid_windVec2
#python plotting_da.py var3d_denseERA5Obs_BINF
#python plotting_da.py var3d_era5B_denseERA5Obs_Binf10_uvwind_1
#python plotting_da.py var3d_NMC12hrB_denseERA5Obs_uvwind
#python plotting_da.py var3d_NMC24hrB_denseERA5Obs_uvwind