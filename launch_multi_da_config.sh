#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=16:00:00
#PBS -q preemptable
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/pytorch_2_troy/
cd /eagle/MDClimSim/mjp5595/ml4dvar/
python run_da_era5.py configs/var3d_era5B_denseERA5Obs_uvwind_0.yml 0 & ;
python run_da_era5.py configs/var3d_era5B_denseERA5Obs_uvwind_1.yml 1 & ;
python run_da_era5.py configs/var3d_NMC12hrB_denseERA5Obs_uvwind.yml 2 & ;
python run_da_era5.py configs/var3d_NMC24hrB_denseERA5Obs_uvwind.yml 3 & ;