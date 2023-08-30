import os.path
import shutil
import subprocess

N_REPEATS = 10

# datasets = ["credit_risk", "credit_score", "bank_marketing", "loan_prediction"]
datasets = ["bank_marketing", "loan_prediction"]
def generate_no_adversary():
    out_dir = 'out_baseline'

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    p = 0
    n_clients = 10
    n_i = -1
    for part_method in ['iid', 'noniid']:
        for data_name in datasets:
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

PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_flr_no_adversary.py --data_name {data_name} --n_repeats {N_REPEATS} --agg_method {agg_method} --percent_adversary {p} --n_clients {n_clients} --part_method {part_method} --n_i {n_i}

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


def generate_baseline():
    out_dir = 'out_baseline'

    # if os.path.exists(out_dir):
    #     shutil.rmtree(out_dir)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    cnt = 0
    for data_name in datasets:
        cnt+=1
        name = f'{data_name}-{N_REPEATS}-baseline'
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
 
PYTHONUNBUFFERED=TRUE python main_lr.py --data_name {data_name} --n_repeats {N_REPEATS}  &> {out_dir}/{data_name}-lr.txt &
PYTHONUNBUFFERED=TRUE python main_dt.py --data_name {data_name} --n_repeats {N_REPEATS}  &> {out_dir}/{data_name}-dt.txt &

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
    cnt += generate_no_adversary()
    cnt += generate_baseline()

    print(f'total number of jobs: {cnt}')
