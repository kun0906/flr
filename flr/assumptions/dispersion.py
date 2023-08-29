import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def deviance_fitted_values():
    # Load a sample dataset (Iris dataset)
    data = load_iris()
    X = data.data
    y = (data.target == 0).astype(int)  # Binary classification: Setosa or not

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a logistic regression model
    X_train = sm.add_constant(X_train)  # Add a constant term for intercept
    logit_model = sm.Logit(y_train, X_train)
    logit_result = logit_model.fit()

    # Calculate deviance residuals
    deviance_residuals = logit_result.resid_response

    # Create Deviance Residuals vs. Fitted Values plot
    plt.scatter(logit_result.fittedvalues, deviance_residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Fitted Values")
    plt.ylabel("Deviance Residuals")
    plt.title("Deviance Residuals vs. Fitted Values")
    plt.tight_layout()
    plt.show()

def variance():
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # Load a sample dataset (Iris dataset)
    data = load_iris()
    X = data.data
    y = (data.target == 0).astype(int)  # Binary classification: Setosa or not

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit a logistic regression model
    X_train = sm.add_constant(X_train)  # Add a constant term for intercept
    logit_model = sm.Logit(y_train, X_train)
    logit_result = logit_model.fit()

    # Calculate deviance residuals
    deviance_residuals = logit_result.resid_response

    # Group deviance residuals by predicted probabilities
    predicted_probs = logit_result.predict(X_train)
    grouped_resids = []
    grouped_probs = []

    # Bin the predicted probabilities to create groups
    bin_edges = np.linspace(0, 1, num=20)  # You can adjust the number of bins as needed
    for i in range(len(bin_edges) - 1):
        mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i + 1])
        if np.any(mask):
            grouped_resids.append(deviance_residuals[mask])
            grouped_probs.append((bin_edges[i] + bin_edges[i + 1]) / 2)

    # Calculate variance for each group of residuals
    variances = [np.var(resids) for resids in grouped_resids]

    # Create Variance Function plot
    plt.plot(grouped_probs, variances, marker='o')
    plt.xlabel("Predicted Probabilities")
    plt.ylabel("Variance of Residuals")
    plt.title("Variance Function Plot")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    variance()