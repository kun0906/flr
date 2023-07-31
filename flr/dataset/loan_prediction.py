"""
    bank_marketing

    dataset:
        https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

import pandas as pd

df = pd.read_csv('dataset/loan_prediction/Training Data.csv', header=0, sep=',')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['Id', 'Married/Single', 'House_Ownership', 'Car_Ownership', 'Profession', 'CITY', "STATE"])
print(f'shape: {df.shape}')
print(df['Risk_Flag'].value_counts())

label = 'Risk_Flag'
mask = df[label]==1
# df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df.loc[mask, label] = '1'
df.loc[~mask, label] = '0'
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv('dataset/loan_prediction/Training_Data-num.csv', index=False)
