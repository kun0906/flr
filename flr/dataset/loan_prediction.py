"""
    Loan prediction

    dataset:
        https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior?select=Training+Data.csv

    shape: (252000, 6)
    Risk_Flag
    0    221004
    1     30996
"""
import os.path

import pandas as pd

file_name = 'dataset/loan_prediction/Training Data.csv'
df = pd.read_csv(file_name, header=0, sep=',')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['Id', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', "STATE"])
print(f'shape: {df.shape}')
print(df['Risk_Flag'].value_counts())

label = 'Risk_Flag'
# mask = df[label]==1
# # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# df.loc[mask, label] = 1
# df.loc[~mask, label] = 0
# classes = {'no':0, 'yes':1}   # positive label
# df[label] = df[label].replace({v:ind for ind, v in enumerate(classes)})
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df = df.loc[:, [v for v in list(df.columns) if v !=label]+[label]]  # move the label to the last column
df.to_csv(os.path.splitext(file_name)[0]+'-num.csv', index=False)
