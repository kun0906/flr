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
N_REPEATS=10
# "bank_marketing" "loan_prediction"  # credit_score
for data_name in "credit_risk" "credit_score"; do
  for p in 0.0 0.05 0.1 0.15 0.2; do  # different attacker percents, e.g., 5%, n = 100, n_normal = 95 clients, n_attackers = 5 clients
    prefix="${data_name}-${p}"
    # for trimmed_mean, we only use 0.1, so here the maximum p is just 0.2
    #  python main_flr.py --data_name $data_name --n_repeats $N_REPEATS --agg_method 'trim_mean' --with_adversary 'true'
    python main_flr.py --data_name $data_name --n_repeats $N_REPEATS --agg_method 'mean' --with_adversary $p &>  ${prefix}-flr_mean.txt &
    python main_flr.py --data_name $data_name --n_repeats $N_REPEATS --agg_method 'median' --with_adversary $p &> ${prefix}-flr_median.txt &
    python main_flr.py --data_name $data_name --n_repeats $N_REPEATS --agg_method 'trim_mean' --with_adversary $p &> ${prefix}-flr_trim_mean.txt &
  done
  python main_lr.py --data_name $data_name --n_repeats $N_REPEATS  &> ${data_name}-lr.txt &
  python main_dt.py --data_name $data_name --n_repeats $N_REPEATS  &> ${data_name}-dt.txt &
done

wait
echo "Finished"

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete

wait
