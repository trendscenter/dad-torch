#!/bin/bash
NUM_SITES=(2 4 8)
BATCH_SIZES=(16 64 128)
EPOCH=101
PHASE="train"
BACKEND="gloo"

for s in "${NUM_SITES[@]}"
do
  for b in "${BATCH_SIZES[@]}"
  do
    python3 MNIST_dadtorch.py -ph $PHASE -b $b -ddp True --dad-reduction False -log net_logs_DDPs"$s"b"$b" -e $EPOCH --world-size $s --dist-backend $BACKEND -gpus
    python3 MNIST_dadtorch.py -ph $PHASE -b $b -ddp True --dad-reduction True -log net_logs_DAD_AGs"$s"b"$b" -e $EPOCH --world-size $s --dist-backend $BACKEND --comm-mode ag --ignore-backward False -gpus
    python3 MNIST_dadtorch.py -ph $PHASE -b $b -ddp True --dad-reduction True -log net_logs_DAD_AG-IBs"$s"b"$b" -e $EPOCH --world-size $s --dist-backend $BACKEND --comm-mode ag --ignore-backward True -gpus
    python3 MNIST_dadtorch.py -ph $PHASE -b $b -ddp True --dad-reduction True -log net_logs_DAD_BCs"$s"b"$b" -e $EPOCH --world-size $s --dist-backend $BACKEND --comm-mode bc --ignore-backward False -gpus
    python3 MNIST_dadtorch.py -ph $PHASE -b $b -ddp True --dad-reduction True -log net_logs_DAD_BC-IBs"$s"b"$b" -e $EPOCH --world-size $s --dist-backend $BACKEND --comm-mode bc --ignore-backward True -gpus
    done
done
