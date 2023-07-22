#!/bin/bash

echo "Starting"
python main_flr.py --n_repeats 5
python main_lr.py --n_repeats 5
python main_dt.py --n_repeats 5

echo "Finished"

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
