#!/bin/bash

initial_directory=$(pwd)

# java 17 installation
if type -p java && [[ $(java -version 2>&1 | awk '/version/ {print $3}' | cut -d'"' -f2) == "17" ]];
then
    echo "Java 17 is already installed."
else
    sudo apt install openjdk-17-jdk -y
fi

# maven installation
if command -v mvn &> /dev/null
then
  echo "Maven is already installed."
else
  sudo apt install -y maven
fi

# python
if command -v python3.8 &>/dev/null;
then
    echo "Python 3.8 is already installed."
else
    sudo apt apt install python3.8 python3.8-venv python3-venv -y
fi

# maven build
mvn clean package

# python venv
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



