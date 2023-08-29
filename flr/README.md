
scp ky8517@nobel.princeton.edu:'/u/ky8517/flr/flr/out/results.pkl' ~/Downloads


baseline: 
    python3 sbatch_baseline.py

various:
    python3 sbatch_various.py
