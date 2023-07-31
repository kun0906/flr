import pandas as pd
import sklearn.metrics
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def load_credit_risk(random_state=42):
    """Loads the Credit Risk dataset from:
    """
    df = pd.read_csv('dataset/legacy/credit_risk_dataset-num.csv')
    X, y = df.values[:, :-1], df.values[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=random_state,
                                                        shuffle=True)
    return (x_train, y_train), (x_test, y_test)


def load_data(data_name='bank_marketing', random_state=42):
    """Loads data from:
    """
    dataname2path = {'bank_marketing':'bank_marketing/bank/bank-full-num.csv',
                 'credit_loan': ''}
    df = pd.read_csv(f'dataset/{dataname2path[data_name]}')
    X, y = df.values[:, :-1], df.values[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=random_state,
                                                        shuffle=True)
    return (x_train, y_train), (x_test, y_test)


def evaluate(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)
    loss = log_loss(y_test, y_probs)
    accuracy = model.score(X_test, y_test)
    auc = roc_auc_score(y_test, y_probs[:, 1])
    return {'loss': loss, "accuracy": accuracy, 'auc': auc}



