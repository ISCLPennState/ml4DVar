#!/bin/bash --login
#PBS -l select=2:system=polaris
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A MDClimSim
#PBS -l filesystems=home:eagle:grand
#PBS -m bae

. /etc/profile

RUNSPERNODE=7

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: {$TSTAMP}"

# Figure out training environment
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    MASTER_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

echo $NNODES
# Commands to run prior to the Python script for setting up the environment
PRELOAD="source /etc/profile ; "
PRELOAD+="ml conda;"
#PRELOAD+="conda activate climaX;"
PRELOAD+="conda activate /eagle/MDClimSim/troyarcomano/.conda/envs/stormer_ace/;"
PRELOAD+="cd /eagle/MDClimSim/mjp5595/ml4dvar/process-obs/ ;"
PRELOAD+="export NODES=1; "

# time python process to ensure timely job exit
TIMER="timeout 718m "

# Launch the pytorch processes on each worker (use ssh for remote nodes)
RANK=0
for NODE in $RANKS; do 
    for ((lNODE=1;lNODE<=$RUNSPERNODE;lNODE++)); do
        echo $lNODE
        if [[ "$NODE" == "$HOSTNAME" ]]; then
            echo "Launching rank $RANK on local node $NODE"
            # Training script and parameters
            CMD="python ERA5_dense_obs.py $RANK"
            FULL_CMD=" $PRELOAD $TIMER $CMD $@ "
            eval $FULL_CMD &
        else
            echo "Launching rank $RANK on remote node $NODE"
            CMD="python ERA5_dense_obs.py $RANK"
            FULL_CMD=" $PRELOAD $TIMER $CMD $@ "
            ssh $NODE "cd $PWD; $FULL_CMD" &
        fi
        RANK=$((RANK+1))
    done
done

wait