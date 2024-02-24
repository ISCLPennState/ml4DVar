#!/bin/bash

#Shell script that submits ml4dvar_pangu jobs to allow for continual cycling using
#Watches if current job is completed and then cycles the job
#qsub -N da_job"$i" -Wblock=true launch_da.sh var4d var4d_zeroBackground

START=1

NUM_JOBS=$3

for i in $(seq $START $NUM_JOBS)
do
    echo starting run_4dvar.py
    echo $i
    qsub -N da_job"$i" -Wblock=true -v da_type=$1,exp_name=$2 launch_da.sh
    #qsub -N da_job"$i" -Wblock=true launch_da.sh
done