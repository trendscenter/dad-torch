#!/bin/bash
NUM_SITES=(2)
BATCH_SIZES=(16 64 128)

PHASE="train"
BACKEND="nccl"

EPOCH=51

#python MNIST_dadtorch.py -ddp True -ph train -nw 4 --dist-backend nccl -b 64 -log net_logs --dad-reduction dad
for b in "${BATCH_SIZES[@]}"
do
  python3 MNIST_dadtorch.py -ddp True -ph $PHASE -nw 4 -e $EPOCH --dist-backend nccl -b $b -log net_logs_dSGD"$b" --dad-reduction base
  python3 MNIST_dadtorch.py -ddp True -ph $PHASE -nw 4 -e $EPOCH --dist-backend nccl -b $b -log net_logs_DAD"$b" --dad-reduction dad
  python3 MNIST_dadtorch.py -ddp True -ph $PHASE -nw 4 -e $EPOCH --dist-backend nccl -b $b -log net_logs_DAD-IB"$b" --dad-reduction dad --ignore-backward True
  python3 runtime_plotter.py -paths net_logs_dSGD"$b" net_logs_DAD"$b" net_logs_DAD-IB"$b" -keys batch_duration -name $b
done

