"""Test randomness of the data


    Runs Test (Wald-Wolfowtiz Test): Checks for the occurrence of runs (consecutive occurrences of the same value) to
    assess if the sequence is random.

"""


def randomness():
    """
    Wald-Wolfowitz test

    Keep in mind that the runs test has assumptions and limitations. It's sensitive to the length of the sequence
    and can be affected by outliers. Additionally, it's recommended to use the runs test as part of a broader analysis
    and consider other diagnostic methods to assess the randomness of your data.

    :return:
    """
    import numpy as np
    from statsmodels.stats.diagnostic import runs

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

randomness()