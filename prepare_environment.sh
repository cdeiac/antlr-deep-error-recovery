#!/bin/bash

initial_directory=$(pwd)

# maven
mvn clean package
# python
cd src/main/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


cd "$initial_directory"
