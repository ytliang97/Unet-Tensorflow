#!/bin/bash

# You should change the output_dir after training.
output_dir="model_output_20200513160656"
phase="test"
testing_set="./datasets/Carvana/test"
# which trained model will be used.
checkpoint="model-10000"

python3 main.py --output_dir=$output_dir \
               --phase=$phase \
               --testing_set=$testing_set \
               --checkpoint=$checkpoint

