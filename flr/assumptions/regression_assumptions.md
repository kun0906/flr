

# Assumptions of Linear Regression:

- Linearity: The relationship between the independent variables and the dependent variable is linear.

- Independence of Residuals: The residuals are independent and do not exhibit autocorrelation.

- Homoscedasticity: The variance of the residuals is constant across all levels of the independent variables.

- Normality of Residuals (why need this?): The residuals follow a normal distribution.
    
- No Multicollinearity: The independent variables are not highly correlated with each other.


# Assumptions of Logistic Regression:

- Linearity of Log Odds: The log odds of the dependent variable are linearly related to the independent variables.

- Independence of Errors: The errors (deviations of observed from predicted values) are independent.

- Large Sample Size: Logistic regression assumes a sufficiently large sample size for reliable parameter estimation.

- No Multicollinearity: Similar to linear regression, logistic regression assumes minimal multicollinearity between independent variables.

- No Perfect Separation: There is no perfect separation of categories by a combination of independent variables. This can lead to unstable coefficient estimates.
    When perfect separation exists, logistic regression may yield extremely large or small parameter estimates. This is because the likelihood function becomes singular, making it difficult to estimate reliable coefficients. This can lead to unrealistic model interpretations and predictions.


It's important to note that while assumptions guide the interpretation and reliability of regression results, models might still be useful even if some assumptions are not perfectly met. 
If assumptions are significantly violated, it might be necessary to transform variables, consider different models, or apply advanced techniques to address the issues.