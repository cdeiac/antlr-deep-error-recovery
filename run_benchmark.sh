#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <data_path> <noise_levels>"
    exit 1
fi

data_path=$1
shift

if [ "$#" -lt 1 ]; then
    echo "Please provide at least one noise level"
    exit 1
fi

noise_levels=("$@")

for noise_level in "${noise_levels[@]}"; do
    # generate noisy dataset
    data_dir=$(./generate_data.sh "$data_path" "$noise_level")
    # train model on noisy data
    ./train_model.sh "$data_dir" False False
    echo "$data_dir"
    # zip and cleanup
    zip -r "src/main/python/logs/$data_dir.zip" "src/main/python/logs/$data_dir/"
    rm -rf "src/main/python/logs/$data_dir/"
    zip -r "src/main/python/data/generated/cache/$data_dir.zip" "src/main/python/data/generated/cache/$data_dir/"
    rm -rf "src/main/python/data/generated/cache/$data_dir/"
    zip -r "src/main/python/data/generated/cache/checkpoints/$data_dir.zip" "src/main/python/data/generated/cache/checkpoints/$data_dir/"
    rm -rf "src/main/python/data/generated/cache/checkpoints/$data_dir/"
    zip -r "src/main/python/data/generated/cv/$data_dir.zip" "src/main/python/data/generated/cv/$data_dir/"
    rm -rf "src/main/python/data/generated/cv/$data_dir/"
done