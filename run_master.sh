#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task 4
#SBATCH -p qTRDGPU

#SBATCH --gpus 1
#SBATCH -A PSYC0002
#SBATCH --oversubscribe
##SBATCH --exclude trendsagn001.rs.gsu.edu,trendsagn002.rs.gsu.edu,trendsagn003.rs.gsu.edu,trendsagn004.rs.gsu.edu,trendsagn005.rs.gsu.edu,trendsagn006.rs.gsu.edu,trendsagn011.rs.gsu.edu,trendsagn012.rs.gsu.edu,trendsagn013.rs.gsu.edu,trendsagn014.rs.gsu.edu,trendsagn015.rs.gsu.edu,trendsagn016.rs.gsu.edu,trendsagn017.rs.gsu.edu,trendsagn018.rs.gsu.edu,trendsagn020.rs.gsu.edu
eval "$(conda shell.bash hook)"

cd /home/users/akhanal1/TrendsLab/dad-torch/
conda activate pytorch_env

rank=$1
mode=$2
sites=$3
branch=$4

pip uninstall dad-torch -y
if [[ "$branch" == "local" ]];
  then
    echo "********** Local installation **********"
    sh ./deploy.sh
else
  echo "********** Git branch installation: "$branch"  *********"
  pip install git+https://github.com/trendscenter/dad-torch.git@$branch
fi

cd examples
python MNIST_dadtorch.py -ddp True --node-rank 0 --num-nodes $sites --dad-reduction $mode --dist-url tcp://10.245.12.102:8998 --master-addr 10.245.12.102 --master-port 8998 -ph train --dist-backend nccl --batch_size 64 -log "net_logs/"$branch"/"$mode"-DADs"$sites
