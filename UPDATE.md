v0.0.6: Reimplement data generation for IID and Non-IID.

1. Reimplement data generated for IID and Non-IID for with/without outliers.
    For IID, each client has the same distribution as the population.
    For Non-IID, each client has the various distributions for different class 
2. Add 'data_partition.py' to include all the data generate functions. 
3. Update the corresponding files.



v0.0.5: Modify FLR for binary tasks

1. Modify FLR for binary tasks with n_class = 1 for params
2. Modify sbatch_*.py



v0.0.4: Run all datasets with selected features by VIF

1. Run all datasets with selected features by VIF
2. Add selected features in load_data()
3. Update sbatch_*.py 



v0.0.3: Implement multi-classification for main_flr 

1. Implement multi-classification for main_flr
2. Add gen_data.sh to generate input data for each case
3. Change 'label' to the last column
4. Add 'f1' in the evaluation metrics
5. Add multiple adversaries and different attacker percentages
6. Add assumptions 



v0.0.2: Fix four datasets 

1. Investigate different datasets and fix the four datasets
2. Add the basic functions to parse each dataset
3. Reorganize the dataset folder


v0.0.1: Initialize project

1. Implement FLR, LR, and DT. 
2. Process 'credit_risk_dataset.csv'
3. Implement basic functions

