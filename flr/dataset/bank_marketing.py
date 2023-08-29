"""
    bank_marketing

    dataset:
        https://archive.ics.uci.edu/dataset/222/bank+marketing

    shape: (45211, 8)
    y
    no     39922
    yes     5289
"""
import os
import pandas as pd

file_name = 'dataset/bank_marketing/bank/bank-full.csv'
df = pd.read_csv(file_name, header=0, sep=';')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])
print(f'shape: {df.shape}')
print(df['y'].value_counts())
label = 'y'
# mask = df[label]=='yes'
# # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# df.loc[mask, label] = 1     # positive label
# df.loc[~mask, label] = 0
classes = {'no':0, 'yes':1}   # positive label
df[label] = df[label].replace({v:ind for ind, v in enumerate(classes)})
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df = df.loc[:, [v for v in list(df.columns) if v !=label]+[label]]  # move the label to the last column
df.to_csv(os.path.splitext(file_name)[0]+'-num.csv', index=False)
