#!/bin/bash

#Shell script that submits ml4dvar_pangu jobs to allow for continual cycling using
#Watches if current job is completed and then cycles the job
#qsub -N da_job"$i" -Wblock=true launch_da.sh var4d var4d_zeroBackground

START=1

NUM_JOBS=$1

for i in $(seq $START $NUM_JOBS)
do
    echo starting launch_climax_forecast.py
    echo $i
    qsub -N climax_forcast_job"$i" -Wblock=true launch_climax_forecast.sh
done