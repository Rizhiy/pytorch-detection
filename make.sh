#!/usr/bin/env bash

# Install dependecies
pip install -r requirements.txt

# Cythonise extensions
cd utils/nms
bash make.sh
cd ../..
