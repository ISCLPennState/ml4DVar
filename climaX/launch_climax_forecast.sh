#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A NeuralDE 
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/pytorch_2_troy
cd /eagle/MDClimSim/mjp5595/ml4dvar/climaX/
python climax_val_forecast.py 0 & python climax_val_forecast.py 1 & python climax_val_forecast.py 2 & python climax_val_forecast.py 3 