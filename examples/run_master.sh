#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsagn001.rs.gsu.edu,trendsagn002.rs.gsu.edu,trendsagn003.rs.gsu.edu,trendsagn004.rs.gsu.edu,trendsagn005.rs.gsu.edu,trendsagn006.rs.gsu.edu,trendsagn007.rs.gsu.edu,trendsagn008.rs.gsu.edu,trendsagn009.rs.gsu.edu,trendsagn011.rs.gsu.edu,trendsagn010.rs.gsu.edu,trendsagn016.rs.gsu.edu,trendsagn017.rs.gsu.edu,trendsagn018.rs.gsu.edu,trendsagn020.rs.gsu.edu


eval "$(conda shell.bash hook)"
conda activate pytorch_env

cd /home/users/akhanal1/TrendsLab/dad-torch/examples
rank=$1
mode=$2
sites=$3
PYTHONPATH=../ python MNIST_dadtorch.py -ddp True --node-rank 0 --num-nodes $sites --dad-reduction $mode --dist-url tcp://10.245.12.98:8998 --master-addr 10.245.12.98 --master-port 8998 -ph train --dist-backend nccl --batch_size 64 -log "examples/net_logs/"$mode"_DADs"$sites
