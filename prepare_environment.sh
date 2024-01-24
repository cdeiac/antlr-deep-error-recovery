#!/bin/bash

initial_directory=$(pwd)

# installation
if command -v mvn &> /dev/null
then
  echo "Maven is already installed."
else
  sudo apt install -y maven
fi

# maven
mvn clean package
# python
cd src/main/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd "$initial_directory"

# create data directories
mkdir src/main/resources/data src/main/resources/generated

mkdir src/main/python/data/ src/main/python/data/generated
mkdir src/main/python/data/generated/cache
mkdir src/main/python/data/generated/checkpoints
mkdir src/main/python/data/generated/cv
mkdir src/main/python/logs



