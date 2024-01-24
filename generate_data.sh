#!/bin/bash

# Check if the required parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <data_path> <probability>"
    exit 1
fi

# Store the parameters in variables
data_path=$1
probability=$2

# noise generation
target_directory=$(java -jar target/antlr-deep-error-recovery-1.0.0-SNAPSHOT-jar-with-dependencies.jar -d "$data_path" -p "$probability")

echo "$target_directory"