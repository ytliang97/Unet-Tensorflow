#!/bin/bash

# You should change the train_logdir after training.
train_logdir='model_output_20200509164910'
phase='test'
test_dataset='/Volumes/Backup/company/DataCenter/Carvana/test'
# which trained model will be used.
checkpoint='model-30'

python3 main.py --train_logdir=$train_logdir \
               --phase=$phase \
               --test_dataset=$test_dataset \

