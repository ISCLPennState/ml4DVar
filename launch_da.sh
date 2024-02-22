#!/bin/bash --login
#PBS -l select=1:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A NeuralDE 
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

ml conda
conda activate climaX
cd /eagle/MDClimSim/mjp5595/ml4dvar/
#python run_da.py var4d
#python run_da.py $1 $2
python run_da.py var4d var4d_defVars