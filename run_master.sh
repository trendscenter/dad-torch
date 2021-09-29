#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
master="10.245.10.98"
eval "$(conda shell.bash hook)"
conda activate pytorch19
cd /data/users2/bbaker/projects/dadtorch3
rank=$1
mode=$2
sites=$3
project=$4
master=`hostname -I | cut -f 1 -d " "`
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/users2/bbaker/bin/miniconda3/lib
PYTHONPATH=. python examples/${project}.py -ddp True --node-rank 0 --num-nodes $sites --dad-reduction $mode --dist-url tcp://${master}:8998 --master-addr ${master} --master-port 8998 -ph train --dist-backend gloo --batch_size 64 -log "net_logs/"$project"_"$mode"_DADs"$sites"b64r5"