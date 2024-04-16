#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=16:00:00
#PBS -q preemptable
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/stormer_ace/
cd /eagle/MDClimSim/mjp5595/ml4dvar/

echo ${PBS_O_PATH}
echo ${PBS_O_WORKDIR}
CMD0="python run_da_era5_from_config.py ${PBS_O_WORKDIR}/configs/var3d_era5B_denseERA5Obs_uvwind_0.yml 0"
CMD1="python run_da_era5_from_config.py ${PBS_O_WORKDIR}/configs/var3d_era5B_denseERA5Obs_uvwind_1.yml 1"
CMD2="python run_da_era5_from_config.py ${PBS_O_WORKDIR}/configs/var3d_NMC12hrB_denseERA5Obs_uvwind.yml 2"
CMD3="python run_da_era5_from_config.py ${PBS_O_WORKDIR}/configs/var3d_NMC24hrB_denseERA5Obs_uvwind.yml 3"
eval $CMD0 & $CMD1 & $CMD2 & $CMD3 