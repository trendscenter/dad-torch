#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
#SBATCH --exclude trendsagn001.rs.gsu.edu,trendsagn002.rs.gsu.edu,trendsagn003.rs.gsu.edu,trendsagn004.rs.gsu.edu,trendsagn005.rs.gsu.edu,trendsagn006.rs.gsu.edu,trendsagn007.rs.gsu.edu,trendsagn008.rs.gsu.edu,trendsagn009.rs.gsu.edu,trendsagn011.rs.gsu.edu,trendsagn010.rs.gsu.edu,trendsagn018.rs.gsu.edu

eval "$(conda shell.bash hook)"
conda activate pytorch19
cd /data/users2/bbaker/projects/dad-torch
rank=$1
mode=$2
sites=$3
actual_batch="1"
echo ACTUAL BATCH IS $actual_batch
PYTHONPATH=. python examples/WIKITEXT_transformer.py -ddp True --node-rank 0 --batch_size $actual_batch --num-nodes $sites --dad-reduction $mode --dist-url tcp://10.245.10.98:8998 --master-addr 10.245.10.98 --master-port 8998 -ph train --dist-backend gloo -log  "net_logs/b512_WIKITEXT_"$mode"_DADs"$sites"b"$actual_batch"r5"