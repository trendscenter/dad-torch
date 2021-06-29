#!/bin/bash
python MNIST_dadtorch.py -ph train -seed 1 -f true -seed-all True -log net_logs -e 101 -gpus 0
python MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction False -log net_logs_DDP -e 101 --ignore-backward False
python MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD -e 101 --ignore-backward False
python MNIST_dadtorch.py -ph train -ddp True -seed 1 -f true -seed-all True --dad-reduction True -log net_logs_DAD_NO_BK -e 101 --ignore-backward True


