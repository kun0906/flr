""" Test normality assumption for a given data.

    Main tests:
        1. Shapiro-Wilk test
        2. K-S test
        3. Q-Q plot
        
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pylab

# Generate example data (replace this with your data)
data = np.random.normal(loc=0, scale=1, size=100)

# Shapiro-Wilk Test
shapiro_statistic, shapiro_p_value = stats.shapiro(data)
print("Shapiro-Wilk Test - p-value:", shapiro_p_value)

# Anderson-Darling Test
ad_statistic, ad_critical_values, ad_significance_levels = stats.anderson(data)
print("Anderson-Darling Test - statistic:", ad_statistic)
print("Anderson-Darling Test - critical values:", ad_critical_values)

# Kolmogorov-Smirnov Test
ks_statistic, ks_p_value = stats.kstest(data, 'norm')
print("Kolmogorov-Smirnov Test - p-value:", ks_p_value)

# Q-Q Plot
sm.qqplot(data, line='s')
plt.title("Q-Q Plot")
plt.show()

# Histogram
plt.hist(data, bins='auto', density=True)
plt.title("Histogram")
plt.show()
