#!/bin/bash

train_logdir='model_output_'`date +'%Y%m%d%H%M%S'`
phase='train'
# training_set='./datasets/apple2orange' # Otherwise, you can use Queue or numpy ndarray to load image.
# tfrecord file
dataset_dir='./datasets/Carvana'
dataset='Carvana_train.tfrecord'
train_batch_size=2
training_number_of_steps=30
summary_steps=100
save_steps=100
checkpoint_steps=2
tf_initial_checkpoint_dir='model_output_20200509163328'

# new training
python3 main.py --train_logdir=$train_logdir \
               --phase=$phase \
               --dataset_dir=$dataset_dir \
               --dataset=$dataset \
               --train_batch_size=$train_batch_size \
               --training_number_of_steps=$training_number_of_steps \
               --summary_steps=$summary_steps \
               --save_steps=$save_steps \
               --checkpoint_steps=$checkpoint_steps \
               --tf_initial_checkpoint_dir=$tf_initial_checkpoint_dir