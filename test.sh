#!/bin/bash

# You should change the output_dir after training.
output_dir="model_output_20200507014719"
phase="test"
testing_set="/Users/yenciliang/Documents/DataCenter/Carvana/test"
# which trained model will be used.
checkpoint="model-8500"

python3 main.py --output_dir="$output_dir" \
               --phase="$phase" \
               --testing_set="$testing_set" \
               --checkpoint="$checkpoint"

