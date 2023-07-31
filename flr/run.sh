#!/bin/bash

echo "Starting"
N_REPEATS=2
python main_flr.py --n_repeats $N_REPEATS --agg_method 'mean' --with_adversary 'true' &>  flr_mean.txt &
python main_flr.py --n_repeats $N_REPEATS --agg_method 'median' --with_adversary 'true' &> flr_median.txt &
python main_lr.py --n_repeats $N_REPEATS  &> lr.txt &
python main_dt.py --n_repeats $N_REPEATS  &> dt.txt &

wait
echo "Finished"

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
