#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe

eval "$(conda shell.bash hook)"
conda activate pytorch_env

python setup.py clean sdist
LATEST_RELEASE="dist/$(ls -t1 dist|  head -n 1)"
TARGET="$1"
pip install $LATEST_RELEASE

cd /home/users/akhanal1/TrendsLab/dad-torch/examples
rank=$1
mode=$2
sites=$3
PYTHONPATH=. python MNIST_dadtorch.py -ddp True --node-rank 0 --num-nodes $sites --dad-reduction $mode --dist-url tcp://10.245.10.78:8998 --master-addr 10.245.10.78 --master-port 8998 -ph train --dist-backend nccl --batch_size 64 -log "net_logs/"$mode"_DADs"$sites
