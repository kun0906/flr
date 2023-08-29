"""
    Credit score classification

    dataset:
        https://www.kaggle.com/datasets/parisrohan/credit-score-classification?select=train.csv

    shape: (100000, 16)
    Credit_Score
    Standard    53174
    Poor        28998
    Good        17828


"""
import os

import pandas as pd

def each_row(row):
    vs = []
    for k in row.index:
        v = row[k]
        if type(v) == int or type(v) == float:
            if k == 'Age':
                v = v if (v > 0 and v < 150) else None
            else:
                if v < 0:
                    v = None
            vs.append(v)
        elif type(v) == str:
            v = v.replace('_', '')
            v = float(v) if v!= '' else None
            if k == 'Age' and v != None and (v< 0 or v>=150):
                v = None
            if k != 'Age' and v != None and (v< 0):
                v = None
            vs.append(v)
        else:
            raise ValueError(v)

    # row = pd.Series(data = [float(v.replace('_', '')) if type(v) == str else v for v in row])
    return pd.Series(vs, index=row.index)

file_name = 'dataset/credit_score/train.csv'
df = pd.read_csv(file_name, header=0, sep=',')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation', 'Monthly_Inhand_Salary', 'Type_of_Loan', 'Credit_Mix', 'Credit_History_Age',
                      'Payment_of_Min_Amount', 'Payment_Behaviour'])
print(f'shape: {df.shape}')
print(df['Credit_Score'].value_counts())

label = 'Credit_Score'
top_classes = ["Poor", "Standard", "Good"]
df = df[df[label].isin(set(top_classes))].reset_index(drop=True)
# transform string to integer
df[label] = df[label].replace({v:ind for ind, v in enumerate(top_classes)}).fillna(value=df[label].mode()).reset_index(drop=True)
df = df.apply(each_row, axis=1)
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df = df.loc[:, [v for v in list(df.columns) if v !=label]+[label]]  # move the label to the last column
# mask = df[label]=='Poor'
# # # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# df.loc[mask, label] = '0'
# mask = df[label]=='Standard'
# df.loc[mask, label] = '1'
# mask = df[label]=='Good'
# df.loc[mask, label] = '2'
# df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv(os.path.splitext(file_name)[0]+'-num.csv', index=False)
