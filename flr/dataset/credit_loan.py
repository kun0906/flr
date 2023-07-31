"""
    Credit risk

    dataset:
        https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset
"""

import pandas as pd

df = pd.read_csv('dataset/credit_loan/loan.csv', header=0, sep=',')
# TODO: Transform categorical features to numerical features
# df = df.drop(columns=['Time'])
print(f'shape: {df.shape}')
print(df['loan_status'].value_counts())
# TODO: Transform categorical features to numerical features
# label = 'loan_status'
# mask = df[label]=='1'
# # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# df.loc[mask, label] = '1'
# df.loc[~mask, label] = '0'
# df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
# df.to_csv('creditcard-num.csv', index=False)
