#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/stormer_ace/
cd /eagle/MDClimSim/mjp5595/ml4dvar/
#python run_da_era5.py $da_type "${exp_name}_gpu0" 0 & python run_da_era5.py $da_type "${exp_name}_gpu1" 1 & python run_da_era5.py $da_type "${exp_name}_gpu2" 2 & python run_da_era5.py $da_type "${exp_name}_gpu3" 3
CMD0="python run_da_era5.py $da_type "var3d_denseERA5Obs_BBHINF" 0"
CMD0="python run_da_era5.py $da_type "var3d_denseERA5Obs_BINF" 1"
#CMD0="python run_da_era5.py $da_type "var3d_NMC12hrB_era5Obs_Jan15" 2"
#CMD0="python run_da_era5.py $da_type "var3d_NMC24hrB_era5Obs_Jan15" 3"

#eval $CMD0 & $CMD1 & $CMD2 & $CMD3 
eval $CMD0 & $CMD1