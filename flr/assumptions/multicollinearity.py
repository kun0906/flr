"""Test multicollinearity

    Main methods:
        1. Correlation Matrix:
        2. Variance Inflation Factor (VIF): 1/(1-R**2)
        3. Tolerance: 1-R**2

    Instruction:
        cd flr
        python3 assumptions/multicollinearity.py
"""
import collections

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from utils import load_data, get_column_names

import matplotlib.pyplot as plt


# def vif(X):
#     """ VIF measures how much the variance of the estimated regression coefficient of a predictor is
#     increased due to multicollinearity with other predictors.
#     High VIF values indicate strong multicollinearity.
#
#     Generally, if a predictor has a VIF greater than 5 or 10, it indicates potential multicollinearity
#     that should be investigated further.
#
#     for each variable f_i,
#     1. we bulid a linear regression with the rest of features and label (y=f_i)
#     2. r_squared = 1 - SSE/SST = 1 - sum(y_i - y_pred)**2/sum(y_i-mu), where mu = mean(f_i)
#     3. vif_i = 1/(1-r_squared)
#
#     :param X:
#     :return:
#     """
#     # X = np.random.normal(loc=0, scale=1, size=100)
#     from statsmodels.stats.outliers_influence import variance_inflation_factor
#     n, d = X.shape
#     res = [0] * d
#     for j in range(d):
#         res[j] = variance_inflation_factor(X, j)  # OLS(X^', Xj), where X^' doesn't include Xj
#
#     print(f"vif: {res}")
#     print(f"vif (sorted): {sorted(res, reverse=True)}")
#     return res
#

def feature_selection_vif(X, vif_threshold=1):
    """
     # Set a threshold for VIF (e.g., 10)
    VIF measures how much the variance of the estimated regression coefficient of a predictor is
    increased due to multicollinearity with other predictors.
    High VIF values indicate strong multicollinearity.

    Generally, if a predictor has a VIF greater than 5 or 10, it indicates potential multicollinearity
    that should be investigated further.

    for each variable f_i,
        1. we bulid a linear regression with the rest of features and label (y=f_i)
        2. r_squared = 1 - SSE/SST = 1 - sum(y_i - y_pred)**2/sum(y_i-mu), where mu = mean(f_i)
        3. vif_i = 1/(1-r_squared)

    :param X:
    :return:
    """
    # Calculate initial VIF values
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    columns = np.asarray([True for i in range(X.shape[1])])

    # Calculate initial VIF values
    vif = np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    max_vif = max(vif)

    # Perform VIF-based feature elimination iteratively
    removed_features = []
    # We at least keep two features.
    while max_vif >= vif_threshold and sum(columns) > 2:
        # Find the variable with the highest VIF
        is_done = True
        for i, (flg, v) in enumerate(zip(columns, vif)):
            if not flg: continue
            if v == max_vif:
                columns[i] = False
                is_done = False
                vif[i] = 0
                removed_features.append(i)
                break
        if is_done: break
        print(max_vif, removed_features, sum(columns), list(vif))
        # Recalculate VIF values for the rest features
        vif = np.zeros((len(columns),))
        j = 0
        for i, flg in enumerate(columns):
            if not flg: continue
            _X = X[:, columns]
            vif[i] = variance_inflation_factor(_X, j)
            j += 1
        max_vif = max(vif)
        # print(max_vif, sum(columns), list(vif))

    # Display the selected variables
    selected_columns = [i for i in range(X.shape[1]) if columns[i] == True]
    print("Selected columns:", selected_columns)

    return selected_columns


def correlation(df, img_file='.png', title=''):
    corr = df.corr()

    plt.figure(figsize=(18, 15))  # width height
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(img_file, dpi=300, pad_inches=0.0)
    plt.show()


if __name__ == '__main__':
    random_state = 42
    for data_name in ['credit_risk']:  # 'credit_score',
        print(f'{int(random_state / 100)}th repeat, and random_state: {random_state}')
        (X_train, y_train), (X_test, y_test) = load_data(data_name, random_state=random_state)
        columns, label = get_column_names(data_name, random_state)
        print(f'X_train: {X_train.shape}, y_train: {collections.Counter(y_train)}')
        print(f'X_test: {X_test.shape}, y_test: {collections.Counter(y_test)}')
        print(columns, label)
        is_normalization = True  # recommended preprocessing
        if is_normalization:
            std = StandardScaler()
            std.fit(X_train)
            X_train = std.transform(X_train)
            X_test = std.transform(X_test)

        img_file = f'out/{data_name}-correlation.png'
        df = pd.DataFrame(X_train, columns=columns)
        correlation(df, img_file, title=data_name)

        # Reduce the redundant features from X_train by VIF
        selected_columns = feature_selection_vif(X_train, vif_threshold=10)
        X_train = X_train[:, selected_columns]
        img_file = f'out/{data_name}-correlation-vif.png'
        df = pd.DataFrame(X_train, columns=[columns[j] for j in selected_columns])
        correlation(df, img_file, title=data_name)
