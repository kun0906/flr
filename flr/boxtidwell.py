import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



def load_data(data_name='bank_marketing', random_state=42, is_selection=True):
    """Loads data from:
    """
    dataname2path = {'bank_marketing':'bank_marketing/bank/bank-full-num.csv',
                     'loan_prediction': 'loan_prediction/Training_Data-num.csv',
                     'credit_score': 'credit_score/train-num.csv',
                      'credit_risk': 'credit_risk/loan-num.csv',
                     }
    df = pd.read_csv(f'dataset/{dataname2path[data_name]}')
    X, y = df.values[:, :-1], df.values[:, -1]
    if is_selection:
        # selected by VIF
        selected_features = {'bank_marketing': [0, 1, 2, 3, 4, 5, 6],
                             'loan_prediction': [0, 1, 2, 3, 4],
                             'credit_score': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                              'credit_risk': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
                             }
        X = X[:, selected_features[data_name]]
    y = y.astype(int)   # change target to integer

    return X,y

def xlog(x):
    return x*np.log(x)


#load data
#X,y=load_data()
X,y=load_data('loan_prediction')

#clean data for non-positive values, so log is well defined
X=np.where(X<=0,0.01,X)

#add log-interaction terms
n=X.shape[1]
Z=np.concatenate((X,xlog(X)),axis=1)

#calculate regression
log_reg=sm.Logit(y,sm.add_constant(Z)).fit()
coefs=log_reg.params
p=log_reg.pvalues

print("Intercept: "+str(coefs[0])+" p-value "+str(p[0]))
for i in range(0,n):
    print("Coef: "+str(coefs[i+1])+" p-value "+str(p[i+1])+" xlog coef: "+str(coefs[i+n+1])+" p-value "+str(p[i+n+1]))
