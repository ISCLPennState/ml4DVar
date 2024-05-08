#!/bin/bash

#Shell script that submits ml4dvar_pangu jobs to allow for continual cycling using
#Watches if current job is completed and then cycles the job
#qsub -N da_job"$i" -Wblock=true launch_da.sh var4d var4d_zeroBackground

START=1

NUM_JOBS=$2

for i in $(seq $START $NUM_JOBS)
do
    echo starting run_4dvar.py
    echo $i
    #qsub -N da_job"$i" -Wblock=true -v da_type=$2,exp_name=$3 $1
    qsub -N da_job"$i" -Wblock=true $1
    #qsub -N da_job"$i" -Wblock=true launch_da.sh
done