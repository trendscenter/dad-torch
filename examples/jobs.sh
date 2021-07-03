#!/bin/bash
#python3 MNIST_dadtorch.py -ph train -seed 1 -f true -seed-all True -log net_logs_ONE -e 51
python3 MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction False -log net_logs_DDP -e 51 --node-rank 0 --world-size 4 --dist-backend gloo
python3 MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD_AG -e 51 --node-rank 0 --world-size 4 --dist-backend gloo --comm-mode ag --ignore-backward False
python3 MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD_AG-IB -e 51 --node-rank 0 --world-size 4 --dist-backend gloo --comm-mode ag --ignore-backward True
python3 MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD_BC -e 51 --node-rank 0 --world-size 4 --dist-backend gloo --comm-mode bc --ignore-backward False
python3 MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD_BC-IB -e 51 --node-rank 0 --world-size 4 --dist-backend gloo --comm-mode bc --ignore-backward True


