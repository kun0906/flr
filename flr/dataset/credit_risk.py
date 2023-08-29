"""
    Credit risk

    dataset:
            https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset

        Index :
        shape: (887379, 32)
        Current                                                601779
        Fully Paid                                             207723
        Charged Off                                             45248
        Late (31-120 days)                                      11591
        Issued                                                   8460
        In Grace Period                                          6253
        Late (16-30 days)                                        2357
        Does not meet the credit policy. Status:Fully Paid       1988
        Default                                                  1219
        Does not meet the credit policy. Status:Charged Off       761

"""
import os.path

import pandas as pd

def missing_stats(df):
    stat = df.isna().sum(axis=0).to_frame(name='missing').reset_index()  # sum() default is axis=0
    # https://stackoverflow.com/questions/17232013/how-to-set-the-pandas-dataframe-data-left-right-alignment
    df.style.set_properties(subset=['index'], **{'text-align': 'right'})
    n, d = df.shape
    stat['missing_percent'] = (stat['missing'] / n * 100).apply(lambda x: float(f'{x:.2f}'))
    stat['total'] = n
    missing_value_stat = stat.sort_values(by='missing_percent', ascending=False)
    # f = 'data/missing_value_stat.csv'
    # missing_value_stat.to_csv(f, sep=',')
    return missing_value_stat

file_name = 'dataset/credit_risk/loan.csv'
df = pd.read_csv(file_name, header=0, sep=',')
# TODO: Transform categorical features to numerical features
df = df.drop(columns=[ # all categorical features
                        'id', 'member_id', 'term', 'grade', 'sub_grade', 'emp_title', 'emp_length',
                      'home_ownership', 'verification_status', 'issue_d','pymnt_plan',
                      'url', 'desc', 'purpose', 'title', 'zip_code', 'addr_state',
                      'earliest_cr_line', 'initial_list_status', 'last_pymnt_d', 'next_pymnt_d',
                      'last_credit_pull_d', 'application_type', 'annual_inc_joint', 'dti_joint',
                      'verification_status_joint',
                      # too much missing valeus: > 51%
                      'il_util', 'mths_since_rcnt_il', 'inq_last_12m', 'open_rv_12m', 'open_acc_6m',
                      'open_il_6m', 'open_il_12m', 'open_il_24m', 'total_bal_il', 'max_bal_bc', 'all_util',
                      'inq_fi', 'total_cu_tl', 'mths_since_last_record', 'mths_since_last_major_derog',
                      'mths_since_last_delinq',
                       # unique features
                       'policy_code',
])
# print(missing_stats(df))
# print(f'shape: {df.shape}')
# print(df['loan_status'].value_counts())
# print(df.info())
label = 'loan_status'
top_classes = ["Current", "Fully Paid", "Charged Off", "Late (31-120 days)", "Issued", "In Grace Period"]
df = df[df[label].isin(set(top_classes))].reset_index(drop=True)
# transform string to integer
df[label] = df[label].replace({v:ind for ind, v in enumerate(top_classes)}).fillna(value=df[label].mode()).reset_index(drop=True)
df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
# d = df.shape[-1]
df = df.loc[:, [v for v in list(df.columns) if v !=label]+[label]]  # move the label to the last column
# print(df['class'].value_counts(sort=True))
print(f'shape: {df.shape}')
print(df['loan_status'].value_counts())

# label = 'loan_status'
# mask = df[label]=='1'
# # df[label][mask] = '1' # chain assigning might not work:https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
# df.loc[mask, label] = '1'
# df.loc[~mask, label] = '0'
# df = df.fillna(value=df.median(axis=0), axis=0).reset_index(drop=True)
df.to_csv(os.path.splitext(file_name)[0]+'-num.csv', index=False)
