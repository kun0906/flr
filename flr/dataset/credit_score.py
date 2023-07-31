"""
    Credit score classification

    dataset:
        https://www.kaggle.com/datasets/parisrohan/credit-score-classification?select=train.csv
"""

import pandas as pd

df = pd.read_csv('dataset/credit_score/train.csv', header=0, sep=',')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Monthly_Inhand_Salary', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age',
                      'Payment_of_Min_Amount', 'Payment_Behaviour'])
print(f'shape: {df.shape}')
print(df['Credit_Score'].value_counts())

label = 'Credit_Score'
mask = df[label]=='Poor'
# # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df.loc[mask, label] = '0'
mask = df[label]=='Standard'
df.loc[mask, label] = '1'
mask = df[label]=='Good'
df.loc[mask, label] = '2'
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv('dataset/credit_score/train-num.csv', index=False)
