"""
    check logistic regression assumptions

"""
import collections

import numpy as np
import pandas as pd
import statsmodels.api as sm
import pylab as py
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import shapiro
from utils import load_data


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def linearity(X, y):

    # Select predictor variable (feature)
    X = X["feature"]

    # Transform the predictor variable using the natural logarithm
    X_transformed = np.log(X)

    # Add constant term to the features matrix
    X_transformed = sm.add_constant(X_transformed)

    # Fit a logistic regression model with the transformed variable
    model = sm.Logit(y, X_transformed)
    result = model.fit()

    # Perform the Box-Tidwell test for linearity
    test_statistic, p_value = result.test_linear("feature_ln")

    print("Box-Tidwell test statistic:", test_statistic)
    print("P-value:", p_value)

    # The test_linear method tests the linearity of the transformed variable in the logistic regression model.
    # If the p-value is significant (typically < 0.05), it suggests that the transformation improves the fit of
    # the model and addresses any nonlinearity.
    return


def durbin_watson(X, y):
    # Interpret the Durbin-Watson statistic:
    # - A value around 2 indicates no significant autocorrelation.
    # - Values below 2 suggest positive autocorrelation (positive serial correlation).
    # - Values above 2 suggest negative autocorrelation (negative serial correlation).

    # Select predictor variables (features)
    # X = data[["feature1", "feature2", ...]]

    # Add constant term to the features matrix
    X = sm.add_constant(X)

    # Fit a regression model
    model = sm.OLS(y, X)  # Replace with sm.Logit() for logistic regression
    result = model.fit()

    # Calculate residuals
    residuals = result.resid

    # Perform Durbin-Watson test
    durbin_watson_statistic = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

    print("Durbin-Watson statistic:", durbin_watson_statistic)

def white():
    """Test the homogenerity of variance


    :return:
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_white

    # Load or create your dataset (replace this with your data)
    data = pd.read_csv("your_data.csv")

    # Select predictor variables (features)
    X = data[["feature1", "feature2", ...]]

    # Add constant term to the features matrix
    X = sm.add_constant(X)

    # Fit a regression model
    model = sm.OLS(y, X)
    result = model.fit()

    # Calculate squared residuals
    squared_residuals = result.resid ** 2

    # Perform the White test
    white_test_statistic, p_value, f_statistic, f_p_value = het_white(squared_residuals, X)

    print("White test statistic:", white_test_statistic)
    print("P-value:", p_value)


def homogenerity_of_residuals():
    # predicted values (x-axis) vs. residuals (y-axis)
    # plot()
    pass

def goldfeldquandt():
    """ Test homogeneity

    :return:
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.compat import lzip
    from statsmodels.stats.diagnostic import het_goldfeldquandt

    # Load or create your dataset (replace this with your data)
    data = pd.read_csv("your_data.csv")

    # Select predictor variable (feature) to split the data
    X = data[["predictor"]]

    # Add constant term to the features matrix
    X = sm.add_constant(X)

    # Fit a regression model
    model = sm.OLS(y, X)
    result = model.fit()

    # Perform the Goldfeld-Quandt test
    f_statistic, p_value, left, right = het_goldfeldquandt(result.resid, X)

    print("Goldfeld-Quandt F-statistic:", f_statistic)
    print("P-value:", p_value)


def randomness():
    """
    Wald-Wolfowitz test

    Keep in mind that the runs test has assumptions and limitations. It's sensitive to the length of the sequence
    and can be affected by outliers. Additionally, it's recommended to use the runs test as part of a broader analysis
    and consider other diagnostic methods to assess the randomness of your data.

    :return:
    """
    import numpy as np
    from scipy.stats import runs

    # Generate example data (replace this with your data)
    data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

    # Perform the runs test
    test_statistic, p_value = runs(data)

    print("Runs test statistic:", test_statistic)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("The sequence shows a significant departure from randomness.")
    else:
        print("The sequence appears to be random.")


def _qqplot(ax, data, dist='norm', line='45', fit=False, title='QQ plot', axis_label=True):
    """Compare the theoretical quantiles of a standard norm dist. to your sample quantiles, so you should
    standardize your data first.

    :param data:
    :param dist:
    :param line:
    :param fit:
    :return:
    """
    sw = shapiro_wilk_test(data)

    # Sort the data
    data_sorted = np.sort(data)

    # Calculate theoretical quantiles
    if dist == 'norm':
        # Probability Point Function: ppf, is the inverse of the CDF.
        theoretical_quantiles = stats.distributions.norm.ppf(np.linspace(0.01, 0.99, len(data_sorted)))
    else:
        raise ValueError(dist)

    # Create the scatter plot
    ax.scatter(theoretical_quantiles, data_sorted, alpha=0.5, s=2)

    # Add reference line if specified
    if line == '45':
        ax.plot(theoretical_quantiles, theoretical_quantiles, color='r', linestyle='--')

    # Fit a line if specified
    if fit:
        slope, intercept = np.polyfit(theoretical_quantiles, data_sorted, 1)
        ax.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, color='g', linestyle=':')

    ax.set_title(f"{title}, SW:{sw:.1f}")
    if axis_label:
        # plt.xlabel("Theoretical Quantiles")
        # plt.ylabel("Sample Quantiles")
        # plt.title(title)
        # # plt.grid()
        # plt.show()
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
    # # ax.axis('equal')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)


def qqplot(X, data_name=''):

    ncols = 5
    n, d = X.shape
    nrows = max(1, d//ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows/ncols*10), sharex=False, sharey=False)    # (width, height)
    axes = axes.reshape((nrows, ncols))
    for j in range(d):
        r, c = divmod(j, ncols)
        axis_label=True if c == 0 else False
        _qqplot(axes[r, c], X[:, j], dist='norm', line='45', fit=True, title=f'{j}th', axis_label=axis_label)
    plt.tight_layout()
    fig.suptitle(data_name)
    plt.show()


def ks_2samp(data1, data2):
    data1 = sorted(data1)
    data2 = sorted(data2)

    n1 = len(data1)
    n2 = len(data2)

    # Compute the Kolmogorov-Smirnov statistic
    ks_statistic = 0
    i, j = 0, 0

    while i < n1 and j < n2:
        d = abs(data1[i] - data2[j])
        if d > ks_statistic:
            ks_statistic = d

        if data1[i] < data2[j]:
            i += 1
        else:
            j += 1

    # Compute the critical value based on sample sizes
    critical_value = 1.36 / ((n1 + n2) ** 0.5)

    # Compute the p-value
    d_plus = ks_statistic
    p_value = 2 * (1 - d_plus * critical_value)

    return ks_statistic, p_value


# # Generate two sample datasets (replace these with your data)
# data1 = [0.2, 0.5, 0.7, 1.1, 1.3]
# data2 = [0.3, 0.6, 0.8, 1.2]
#
# # Perform the two-sample KS test
# statistic, p_value = ks_2samp(data1, data2)
#
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# if p_value < 0.05:
#     print("The two samples likely come from different distributions.")
# else:
#     print("The two samples likely come from the same distribution.")
#


def ks(data1, data2):
    import numpy as np
    from scipy.stats import ks_2samp

    # # Sample data (replace this with your actual data)
    data2 = np.random.normal(loc=0, scale=1, size=100)

    # Perform Kolmogorov-Smirnov test
    # The second argument is the name of the theoretical distribution to test against.
    # In this example, we are testing against the standard normal distribution.
    ks_statistic, p_value = ks_2samp(data1, data2)

    # Print the results
    print("KS Statistic:", ks_statistic)
    print("P-value:", p_value)

    # Check the null hypothesis: if p-value is less than the significance level (e.g., 0.05),
    # we reject the null hypothesis, meaning the data does not follow the normal distribution.
    if p_value < 0.05:
        print("The data does not follow the normal distribution.")
    else:
        print("The data follows the normal distribution.")


def shapiro_wilk_test(data):

    # Perform the Shapiro-Wilk test
    statistic, p_value = shapiro(data)

    print("Shapiro-Wilk test statistic:", statistic)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("The data is not normally distributed.")
    else:
        print("The data is normally distributed.")

    return statistic


def vif(X):
    """ VIF measures how much the variance of the estimated regression coefficient of a predictor is
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
    # X = np.random.normal(loc=0, scale=1, size=100)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    n, d = X.shape
    res = [0] * d
    for j in range(d):
        res[j] = variance_inflation_factor(X, j)        # OLS(X^', Xj), where X^' doesn't include Xj

    print(f"vif: {res}")
    print(f"vif (sorted): {sorted(res, reverse=True)}")
    return res


def feature_selection_vif(X):
    """
    Not finished
    :param X:
    :return:
    """
    # Calculate initial VIF values
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    columns = [f'f_{i}' for i in range(X.shape[1])]

    # Calculate initial VIF values
    vif = np.array([(variance_inflation_factor(X, i), i, columns[i]) for i in range(X.shape[1])])
    max_vif = max([v for v, feat in vif])
    # Set a threshold for VIF (e.g., 10)
    vif_threshold = 10

    removed_feats = []
    # Perform VIF-based feature elimination iteratively
    while vif.max() > vif_threshold:
        # Find the variable with the highest VIF
        max_vif_index, feat = [(i, feat) for v, i, feat in vif if v == max_vif][0]
        # Remove the variable with the highest VIF
        X = np.delete(X, max_vif_index, axis=1)
        removed_feats.append(feat)

        # Recalculate VIF values
        vif = np.array([(variance_inflation_factor(X, i), i) for i in range(X.shape[1])])
        max_vif = max([v for v, i in vif])

    # Display the selected variables
    selected_variables = list(range(X.shape[1]))
    print("Selected variables:", selected_variables)

    return selected_variables


if __name__ == '__main__':
    random_state=42
    for data_name in ['credit_score', 'credit_risk']:
        print(f'{int(random_state / 100)}th repeat, and random_state: {random_state}')
        (X_train, y_train), (X_test, y_test) = load_data(data_name, random_state=random_state)
        print(f'X_train: {X_train.shape}, y_train: {collections.Counter(y_train)}')
        print(f'X_test: {X_test.shape}, y_test: {collections.Counter(y_test)}')
        is_normalization = True     # recommended preprocessing
        if is_normalization:
            std = StandardScaler()
            std.fit(X_train)
            X_train = std.transform(X_train)
            X_test = std.transform(X_test)

        # selected_variables = feature_selection_vif(X_train)

        # # Perform the Shapiro-Wilk test
        # test_statistic = [shapiro_wilk_test(X_train[:, j]) for j in range(X_train.shape[1])]
        # print("Shapiro-Wilk test statistic:", test_statistic)
        qqplot(X_train, data_name)
        # ks()
        # vif(X_train)






