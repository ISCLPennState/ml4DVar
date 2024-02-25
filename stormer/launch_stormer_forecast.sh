#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/pytorch_2_troy/
cd /eagle/MDClimSim/mjp5595/ml4dvar/stormer/
python stormer_val_forecast.py 0 & python stormer_val_forecast.py 1 & python stormer_val_forecast.py 2 & python stormer_val_forecast.py 3 