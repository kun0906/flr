import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.metrics import log_loss, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


# def load_credit_risk(random_state=42):
#     """Loads the Credit Risk dataset from:
#     """
#     df = pd.read_csv('dataset/legacy/credit_risk_dataset-num.csv')
#     X, y = df.values[:, :-1], df.values[:, -1]
#     x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
#                                                         random_state=random_state,
#                                                         shuffle=True)
#     return (x_train, y_train), (x_test, y_test)

def sample(X, y, each_size=100000, random_state=42):
    # Assuming X and y are your numpy arrays
    unique_classes = np.unique(y)
    subset_indices = []

    rng = np.random.RandomState(seed=random_state)
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        # Randomly shuffle the indices
        subset_indices.extend(rng.permutation(class_indices)[:each_size])

    subset_X = X[subset_indices]
    subset_y = y[subset_indices]

    return subset_X, subset_y

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
    # X, y = sample(X, y, each_size=10000, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=random_state,
                                                        shuffle=True)
    return (x_train, y_train), (x_test, y_test)


def get_column_names(data_name='bank_marketing', random_state=42):
    """Loads data from:
    """
    dataname2path = {'bank_marketing':'bank_marketing/bank/bank-full-num.csv',
                     'loan_prediction': 'loan_prediction/Training_Data-num.csv',
                     'credit_score': 'credit_score/train-num.csv',
                      'credit_risk': 'credit_risk/loan-num.csv',
                     }
    df = pd.read_csv(f'dataset/{dataname2path[data_name]}')
    # X, y = df.values[:, :-1], df.values[:, -1]
    features, label = df.columns[:-1].tolist(), df.columns[-1]
    return features, label


def check_nan(y_probs, X_test, y_test):
    # Find rows containing NaN values
    rows_with_nan = np.any(np.isnan(y_probs), axis=1)   # 0/0 can lead Nan
    n_nans = sum(rows_with_nan)
    if n_nans > 0:
        print('Total number of NAN rows', n_nans)
        # Print rows containing NaN values
        for row_idx, contains_nan in enumerate(rows_with_nan):
            if contains_nan:
                print(f"Row {row_idx}: {y_probs[row_idx]}, {list(X_test[row_idx])}, {(y_test[row_idx])}")
                return True

    return False

def fill_nan(data):
    # Calculate median of each column excluding NaN
    median_values = np.nanmedian(data, axis=0)

    # Replace NaN values with column medians
    filled_data = np.where(np.isnan(data), median_values, data)
    return filled_data/np.sum(filled_data, axis=1).reshape((filled_data.shape[0], -1))   # normalize the vector

def evaluate(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)
    if check_nan(y_probs, X_test, y_test):
        y_probs = fill_nan(y_probs)  # too large or small input values could lead Nan
        # print(y_probs[88198], sum(y_probs[88198]))    # for debugging
    loss = log_loss(y_test, y_probs)
    accuracy = model.score(X_test, y_test)
    y_preds = model.predict(X_test) # predicted labels
    if len(set(y_test)) == 2:
        f1 = f1_score(y_test, y_preds, average='binary')
        multi_class = 'raise'
        auc = roc_auc_score(y_test, y_probs[:, 1], multi_class=multi_class)
    else:  # one vs rest
        f1 = f1_score(y_test, y_preds, average='macro')
        multi_class = 'ovr'
        auc = roc_auc_score(y_test, y_probs, multi_class=multi_class)
    return {'loss': loss, "accuracy": accuracy, 'f1': f1, 'auc': auc}



