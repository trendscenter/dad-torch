#!/bin/bash
BATCH_SIZES=(64)

PHASE="train"
BACKEND="gloo"

EPOCH=5
for b in "${BATCH_SIZES[@]}"
do
  python3 MNIST_dadtorch.py -ddp True -ph $PHASE -nw 4 -e $EPOCH --dist-backend $BACKEND -b $b -log net_logs_dSGD"$b" --dad-reduction base --world-size 4
  python3 MNIST_dadtorch.py -ddp True -ph $PHASE -nw 4 -e $EPOCH --dist-backend $BACKEND -b $b -log net_logs_DAD"$b" --dad-reduction dad --world-size 4
  python3 runtime_plotter.py -paths net_logs_dSGD"$b" net_logs_DAD"$b" -keys batch_duration -name $b
done

