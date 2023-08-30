"""Test linearity

    Hosmer-Lemeshow Test: if model fit well, it means there is a linearity between logit(P) and Xs.
    Box-Tidwell Test
    Rainbow Test: This test assesses linearity by fitting the regression model and testing whether a linear model is adequate.
    It examines whether the relationship between the residuals and the fitted values is linear.
    RESET Test: Ramsey's RESET test examines whether adding transformed versions of the independent variables improves the fit of the model, suggesting possible non-linearity.

"""
import collections

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from utils import load_data, get_column_names

import matplotlib.pyplot as plt



import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_rainbow
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from statsmodels.stats.diagnostic import linear_rainbow
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import OLSInfluence


def linearity(df, img_file, title=''):
    # # Load or create your dataset (replace this with your data)
    # data = pd.read_csv("dataset/bank_marketing/bank/bank-full-num.csv")
    # y = df['y']
    # y = (df.y == 0).astype(int)  # Binary classification: y or not. one vs rest: changing multiclass to binary
    formula = 'y ~ ' + '+'.join(df.columns[:-1])
    # formula = 'y ~ age+balance+day+duration+campaign+pdays+previous'
    model = smf.logit(formula, df)
    result = model.fit()
    print(result.summary())

    logit = result.predict(df.iloc[:,:-1])

    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    for i in range(df.shape[1]):
        x = df.values[:, i]
        # Calculate the log-odds for the training set
        i, j = divmod(i, 4)
        axes[i, j].scatter(x, logit)
        axes[i, j].set_xlabel(f"{i}")
        axes[i, j].set_ylabel('logit')
    plt.show()


    # # Scatter plot
    # plt.scatter(data["age"], data["y"])
    # plt.title("Scatter Plot")
    # plt.show()
    #
    # # Residual plot
    # plt.scatter(result.fittedvalues, result.resid_response)
    # plt.axhline(y=0, color='r', linestyle='--')
    # plt.title("Residual Plot")
    # plt.show()

    # # Rainbow Test for linearity
    # rainbow_statistic, rainbow_p_value = linear_rainbow(result)
    # print("Rainbow Test - p-value:", rainbow_p_value)
    #

#
# def linearity2(df, img_file, title=''):
#
#     # Fit a logistic regression model
#     # X_train = sm.add_constant(X_train)  # Add a constant term for intercept
#     formula = 'y ~ ' + '+'.join(df.columns[:-1])
#     logit_model = sm.Logit(formula, df)
#     logit_result = logit_model.fit()
#
#     # Calculate squared terms for predictors
#     X = df.values[:, :-1]
#     X_squared = X[:, 1:] ** 2  # Exclude the constant term
#
#     # Add squared terms to the model
#     y = df['y'].values
#     X_train_extended = np.column_stack((X, X_squared))
#     logit_model_extended = sm.Logit(y, X_train_extended)
#     logit_result_extended = logit_model_extended.fit()
#
#     # Calculate residuals from the extended model
#     ols_residuals = logit_result_extended.resid_response
#
#     # Fit an auxiliary OLS regression for residuals against the squared terms
#     ols_model = sm.OLS(ols_residuals, X_squared)
#     ols_result = ols_model.fit()
#
#     # Obtain the auxiliary OLS coefficients
#     auxiliary_params = ols_result.params
#
#     # Perform the Box-Tidwell test
#     box_tidwell_test = logit_result.wald_test(auxiliary_params[1:])
#     print(box_tidwell_test)
#
#     # Interpret the test results
#     alpha = 0.05
#     if box_tidwell_test.pvalue < alpha:
#         print("Reject the null hypothesis. The model might have non-linearity.")
#     else:
#         print("Fail to reject the null hypothesis. The model appears to have linearity.")


def hosmer_lemeshow_test(X_train, y_train):
    """
    The Hosmer-Lemeshow test is a statistical test used to assess the goodness-of-fit of a logistic regression model.

    :return:
    """
    from scipy.stats import chi2

    # Fit a logistic regression model
    X_train = sm.add_constant(X_train)  # Add a constant term for intercept
    logit_model = sm.Logit(y_train, X_train)
    logit_result = logit_model.fit()

    # Calculate predicted probabilities
    predicted_probs = logit_result.predict(X_train)

    # Create bins for the Hosmer-Lemeshow test
    num_bins = 10
    # The observed frequencies represent the actual number of positive outcomes (1s) in each bin of predicted probabilities.
    observed_freq = np.histogram(predicted_probs, bins=num_bins, weights=y_train)[0]
    # Calculate expected frequencies based on predicted probabilities
    density, bin_edges = np.histogram(predicted_probs, bins=num_bins, density=True)
    expected_freq = density * y_train.sum()

    # Calculate the Chi-squared statistic
    chi_squared = ((observed_freq - expected_freq) ** 2 / expected_freq).sum()

    # Calculate the degrees of freedom
    df = num_bins - 2  # Number of bins minus the parameters estimated in the model

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_squared, df)

    print("Chi-squared:", chi_squared)
    print("Degrees of Freedom:", df)
    print("p-value:", p_value)

    # Interpret the results. H0: the model fits well.
    """
    H0: There is no significant difference between the observed and expected frequencies of outcomes within different groups 
    or bins created based on the predicted probabilities from the logistic regression model.
    """
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. The model doesn't fit well.")
    else:
        print("Fail to reject the null hypothesis. The model fits well.")


if __name__ == '__main__':
    random_state = 42
    for data_name in ['bank_marketing']:  # 'credit_score', credit_risk, bank_marketing, loan_prediction
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

        img_file = f'out/{data_name}-linearity.png'
        df = pd.DataFrame(np.concatenate([X_train, y_train.reshape((-1, 1))], axis=1), columns=columns +['y'])
        df['y'] = (y_train == 0).astype(int)    # 0 or not.
        """
        The "Singular matrix" error in the context of logistic regression can occur when the matrix 
        of predictors (independent variables) has multicollinearity, meaning that some of the predictor 
        variables are linearly dependent on each other. 
        """
        # linearity(df, img_file, title=data_name)
        hosmer_lemeshow_test(X_train, df['y'])

