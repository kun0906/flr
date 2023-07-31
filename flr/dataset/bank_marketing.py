"""
    bank_marketing

    dataset:
        https://archive.ics.uci.edu/dataset/222/bank+marketing
"""

import pandas as pd

df = pd.read_csv('dataset/bank_marketing/bank/bank-full.csv', header=0, sep=';')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
print(f'shape: {df.shape}')
print(df['y'].value_counts())
label = 'y'
mask = df[label]=='yes'
# df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df.loc[mask, label] = '1'
df.loc[~mask, label] = '0'
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv('dataset/bank_marketing/bank/bank-full-num.csv', index=False)
