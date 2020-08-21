#!/bin/bash

#SBATCH --gres=gpu:p100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=3   # maximum CPU cores per GPU request: 6 on Cedar, 16$
#SBATCH --mem=12000M        # memory per node
#SBATCH --time=0-08:00      # time (DD-HH:MM)
#SBATCH --output=output.out  # %N for node name, %j for jobID

python3 estimator.py \
  --data_dir=data/ \
  --model_dir=model/ \
  --encoded_data_dir=encoded_data/ \
  --data_name=raw_data \
  --train_steps=100000 \
  --vocab_level=15 \
  --dropout=0.1 \
  --heads=8 \
  --abstract_len=512 \
  --title_len=60 \
  --batch_size=16 \
  --layers=4 \
  --depth=256 \
  --feedforward=512 \
  --train=True \
  --predict=True \
  --predict_samples=10 \
  --description="Put the experimental description here" \

  $@