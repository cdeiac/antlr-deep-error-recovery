#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <data_dir> <load_cv> <load_checkpoint>"
    exit 1
fi

# Store the parameters in variables
data_dir=$1
load_cv=$2
load_checkpoint=$3

# data preparation and training
source src/main/python/venv/bin/activate
python3 src/main/python/main.py --data_dir=$data_dir --load_cv=$load_cv --load_checkpoint=$load_checkpoint
