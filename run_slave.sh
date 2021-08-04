#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe

eval "$(conda shell.bash hook)"
conda activate pytorch19
cd /data/users2/bbaker/projects/dad-torch
rank=$1
mode=$2
sites=$3
PYTHONPATH=. python examples/MNIST_dadtorch.py -ddp True --node-rank $rank --dad-reduction $mode --num-nodes $sites --dist-url tcp://10.245.10.78:8998 --master-addr 10.245.10.78 --master-port 8998 -ph train --dist-backend gloo --batch_size 64 -log "net_logs/"$mode"_DADs"$sites"b64r5"

