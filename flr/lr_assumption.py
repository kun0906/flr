"""
    check logistic regression assumptions

"""

import numpy as np
import statsmodels.api as sm
import pylab as py

def qq_plot(residuals=None):
    # for univariate
    # np.random generates different random numbers
    # whenever the code is executed
    # Note: When you execute the same code
    # the graph look different than shown below.

    # Random data points generated
    data_points = np.random.normal(0, 1, 100)

    sm.qqplot(data_points, line='45')
    py.show()

def ks():
    import numpy as np
    from scipy.stats import kstest

    # Sample data (replace this with your actual data)
    data = np.random.normal(loc=0, scale=1, size=100)

    # Perform Kolmogorov-Smirnov test
    # The second argument is the name of the theoretical distribution to test against.
    # In this example, we are testing against the standard normal distribution.
    ks_statistic, p_value = kstest(data, 'norm')

    # Print the results
    print("KS Statistic:", ks_statistic)
    print("P-value:", p_value)

    # Check the null hypothesis: if p-value is less than the significance level (e.g., 0.05),
    # we reject the null hypothesis, meaning the data does not follow the normal distribution.
    if p_value < 0.05:
        print("The data does not follow the normal distribution.")
    else:
        print("The data follows the normal distribution.")

def vif(X):
    X = np.random.normal(loc=0, scale=1, size=100)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    n, d = X.shape
    res = [0] * d
    for j in range(d):
        res[j] = variance_inflation_factor(X, j)

    print(res)
    return res



if __name__ == '__main__':
    # qq_plot()
    # ks()
    vif()





