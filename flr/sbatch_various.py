import os.path
import shutil
import subprocess


N_REPEATS=10
datasets = ["credit_risk", "credit_score", "bank_marketing", "loan_prediction"]
# datasets = ["credit_risk"]
out_dir = 'out/various'
def generate_fixed_n_sh(n_clients=50, n_i=100):

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    for part_method in ['iid', 'noniid']:
        for data_name in datasets:
            for p in [0.0, 0.05, 0.1, 0.15, 0.2]:
                for agg_method in ['mean', 'median', 'trim_mean']:
                    name = f"{data_name}-{n_clients}-{n_i}-{p}-{agg_method}-{part_method}"
                    cnt+=1
                    s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/flr/flr
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_flr python=3.10.4
conda activate py3104_flr

pwd
python3 -V
uname -a 
hostname -s

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_flr.py --data_name {data_name} --n_repeats {N_REPEATS} --agg_method {agg_method} --percent_adversary {p} --n_clients {n_clients} --part_method {part_method} --n_i {n_i}

# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    """
                    out_sh = f'{out_dir}/{name}.sh'
                    # print(out_sh)
                    with open(out_sh, 'w') as f:
                        f.write(s)

                    cmd = f'sbatch {out_sh}'
                    ret = subprocess.run(cmd, shell=True)
                    print(cmd, ret)
            
    return cnt


def generate_fixed_p_sh(p=0.1, n_i=100):
    # out_dir = 'sh'
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    for part_method in ['iid', 'noniid']:
        for data_name in datasets:
            for n_clients in [10, 50, 100, 150, 200]:  # each client has 100 points per class
                for agg_method in ['mean', 'median', 'trim_mean']:
                    name = f"{data_name}-{n_clients}-{n_i}-{p}-{agg_method}-{part_method}"
                    cnt += 1
                    s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/flr/flr
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_flr python=3.10.4
conda activate py3104_flr

pwd
python3 -V
uname -a 
hostname -s

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_flr.py --data_name {data_name} --n_repeats {N_REPEATS} --agg_method {agg_method} --percent_adversary {p} --n_clients {n_clients} --part_method {part_method} --n_i {n_i}

# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    """
                    out_sh = f'{out_dir}/{name}.sh'
                    # print(out_sh)
                    with open(out_sh, 'w') as f:
                        f.write(s)

                    cmd = f'sbatch {out_sh}'
                    ret = subprocess.run(cmd, shell=True)
                    print(cmd, ret)

    return cnt


def generate_varied_ni_sh(p=0.1, n_clients=50):
    # out_dir = 'sh'
    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)
    #
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    cnt = 0
    for part_method in ['iid', 'noniid']:
        for data_name in datasets:
            for n_i in [25, 50, 100, 150, 200]:
                for agg_method in ['mean', 'median', 'trim_mean']:
                    name = f"{data_name}-{n_clients}-{n_i}-{p}-{agg_method}-{part_method}"
                    cnt += 1
                    s = fr"""#!/bin/bash

#SBATCH --job-name={name}         # create a short name for your job
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={out_dir}/out_{name}.txt
#SBATCH --error={out_dir}/err_{name}.txt

module purge
cd /scratch/gpfs/ky8517/flr/flr
module load anaconda3/2021.11
#conda env list
#conda create --name py3104_flr python=3.10.4
conda activate py3104_flr

pwd
python3 -V
uname -a 
hostname -s

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_flr.py --data_name {data_name} --n_repeats {N_REPEATS} --agg_method {agg_method} --percent_adversary {p} --n_clients {n_clients} --part_method {part_method} --n_i {n_i}

# if you use & at the end of your command, your job cannot be seen by 'squeue -u'

wait
echo 'done'     
    """
                    out_sh = f'{out_dir}/{name}.sh'
                    # print(out_sh)
                    with open(out_sh, 'w') as f:
                        f.write(s)

                    cmd = f'sbatch {out_sh}'
                    ret = subprocess.run(cmd, shell=True)
                    print(cmd, ret)

    return cnt


if __name__ == '__main__':
    cnt = 0
    for n_clients in [100]:  # various p, fixed n, and n_i=100
        cnt += generate_fixed_n_sh(n_clients, n_i=100)
        print(f'\n***total submitted jobs (p, n_i=100, n_clients={n_clients}): {cnt}')

    for p in [0.1]:    # various n, and each client has 10% noises
        cnt += generate_fixed_p_sh(p, n_i=100)
        print(f'\n***total submitted jobs (p={p}, n_i=100, n_clients): {cnt}')

    for p in [0.1]:  # various n_i, fixed p=0.1, n = 100
        cnt += generate_varied_ni_sh(p, n_clients=100)
        print(f'\n***total submitted jobs (p={p}, n_i, n_clients=50): {cnt}')

    print(f'total number of jobs: {cnt}')
