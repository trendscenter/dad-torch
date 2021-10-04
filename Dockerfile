FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
RUN apt-get update
RUN apt-get install -y git
RUN mkdir /app
WORKDIR /app 
RUN git clone https://github.com/trendscenter/dad-torch.git
WORKDIR /app/dad-torch
RUN git checkout aws-debug-ak
RUN git pull
RUN pip install -r requirements.txt
RUN pip install awscli
RUN pip install kaggle
WORKDIR /app/dad-torch
RUN python /app/dad-torch/data/mnist.py 
#RUN kaggle 
ADD master_aws.sh /app/dad-torch/master_aws.sh
ADD slave_aws.sh /app/dad-torch/slave_aws.sh


