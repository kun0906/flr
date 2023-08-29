#!/bin/bash

## 1. Load (/create) and activate python3
#module load anaconda3/2021.11
## 2. Create Python3, ignore it if you already have the environment
#conda create --name py3104_ifca python=3.10.4
#conda activate py3104_ifca

#cd /u/ky8517/flr/flr
#module load anaconda3/2021.11
#conda activate py3104_ifca
# pip install -r requirements.txt


echo "Starting"

PYTHONPATH='.' PYTHONUNBUFFERED=TRUE python3 assumptions/multicollinearity.py
#PYTHONPATH='.' PYTHONUNBUFFERED=TRUE python3 assumptions/linearity.py

wait
echo "Finished"

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete

wait
