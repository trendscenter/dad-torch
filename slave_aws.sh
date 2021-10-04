#!/bin/bash
project=$1
dad_reduction=$2
dist_backend=$3
batch_size=$4
fold=$5
sites=$AWS_BATCH_JOB_NUM_NODES
rank=$AWS_BATCH_JOB_NODE_INDEX
#git checkout aws-debug-ak
#pip install -r requirements.txt
#pip install awscli
log_folder="p${project}_dr${dad_reduction}_be${dist_backend}_ba${batch_size}_kf${fold}_si${sites}"
#hostname_file=${log_folder}"_hostname.txt"
#while true
#do
#    exists=$(aws s3 ls s3://dad-io/hosts/${hostname_file})
#    if [ -z "$exists" ]; then
#    echo "it does not exist"
#    else
#        aw3 s3 cp s3://dad-io/hosts/${hostname_file} ${hostname_file}
#        master=`cat ${hostname_file}`
#        break
#    fi
#done
master=$AWS_BATCH_JOB_MAIN_NODE_INDEX
PYTHONPATH=. python examples/$project -ddp True --dad-reduction $dad_reduction -ph train --dist-backend $dist_backend --batch_size $batch_size -nf 10 --fold-num $fold -log $log_folder  --dist-url tcp://${master}:8998 --master-addr ${master} --master-port 8998 --rank ${rank}
#mkdir to_S3
#mv $log_folder to_S3/ -v
#aws s3 cp to_S3 s3://dad-io/ --recursive