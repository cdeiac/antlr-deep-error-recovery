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
