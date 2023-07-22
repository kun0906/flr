


import pandas as pd

df = pd.read_csv('credit_risk_dataset.csv')
df = df.drop(columns=['person_home_ownership', 'loan_grade', 'loan_intent', 'cb_person_cred_hist_length'])
label = 'cb_person_default_on_file'
mask = df[label]=='Y'
# df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
df.loc[mask, label] = '1'
df.loc[~mask, label] = '0'
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv('credit_risk_dataset-num.csv', index=False)
