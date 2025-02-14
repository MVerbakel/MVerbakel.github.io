---
layout: post
title: "Linear regression (introduction)"
mathjax: true
---

Linear regression is a model for predicting a continuous outcome (a dependent variable), from a set of predictors (independent variable/s). For example, imagine you want to predict the sale price for houses in your neighbourhood. You've gathered data on the houses sold in the past few years, including the actual sale price and some features about the houses (e.g. the number of bedrooms, number of bathrooms, size of the house, and whether it had a pool). From these past sales, we can use the associations between these features and the outcome to fit a function for predicting the price of a new house on the market. While the true relationships between these feature/s and the outcome in the real world are likely complex, we can often get a reasonable approximation with a simple additive linear model (linear regression). Given it's simplicity, efficiency and interpretability, it's one of the most commonly used in practice.

## Intuition

To first build some intuition, imagine we have just one predictor, the distance between the house and the nearest public transport. As demonstrated in the plot below, there is a linear relationship between them: as the distance increases, so to does the sale price (i.e. they have a positive correlation). To predict the sale price of a house, we fit a line through the center of these points, choosing the placement that minimises the distance between the actual sale prices and the predicted values (i.e. the vertical distance from each point to the line). In other words, we try minimise the error between our predictions and the actual values. 

![Simple linear model](/assets/simple_linear_model.png)

When we have multiple features, we're doing the same thing but in a multi-dimensional space. Instead of a line, we fit a hyperplane. This is why we use the term "linear", as we're only using lines and planes (i.e. linear functions of the inputs).

A reminder, this doesn't prove that there is a causal relationship between these variables. The correlation between the distance from public transport and the sale price could be partially (or even fully) explained by other factors (e.g. whether there's a school nearby, or distance from the CBD). 

## Ordinary Least Squares (OLS)

One of the most common methods of fitting this model is OLS (least squares). OLS minimises the sum of the squared residuals (differences between predicted Y and actual Y). You can think of each row in the data set as an equation. Given each of the feature values (X's), we want to determine a set of weights (coefficients or $$\beta$$), that when applied to the feature inputs, gives us the actual sale price (Y), or the closest value possible. We can put this into the following function to solve:

$$\hat{Y} = \beta_0 + \beta_1 * X_1 + ... \beta_n * X_n$$

$$Y = \beta_0 + \beta_1 * X_1 + ... \beta_n * X_n + \epsilon$$

$$\epsilon_i = y_i - \hat{y_i}$$

Where $$\beta_0$$ is the intercept (value of y when x=0), $$\beta_n$$ are the weights for each feature $$X_n$$, and $$\epsilon$$ is the remaining error or unexplained variability in Y (residuals). To find the unknowns, we minimise the sum of the squared residuals, where the residuals are the distances between the actual and predicted sale price:

$$RSS = \epsilon_1^2 + ... + \epsilon_n^2 $$

The errors are squared to avoid positive and negative errors cancelling out, but also to put more emphasis on larger errors. This also provides a smooth and continuous function that can be minimised with different optimisation methods.

### Linear Algebra

Let's first think about how we would solve this in an ideal case with 0 error. If we consider the data set is just a matrix, you may recall from linear algebra that a system of linear equations like this can be solved with elimination. We simplify the equations by subtracting multiples of rows from each other (noting this doesn't change the relationship between the X's and Y as we're doing the same to both sides), until we have an upper triangular matrix. At that point, the last row has only one unknown, which we can easily solve knowing the output. Then we just work backwards up the triangle to solve the remaining, as each time there will be only one unknown (having solved the others). 

However, elimination is only possible when the outcome Y is a linear combination of the inputs X. Geometrically, this would mean Y is in the column space of X (the column space contains all linear combinations of the column vectors). Further, the rows and columns must be independent to produce only one solution (i.e. we require a square invertible matrix). It's more common that we have a tall and thin matrix (many observations, but only a few features), in which case there will be either 0 or 1 solution. If the matrix is short and wide (many more features than observations), some features are free, meaning they can take any value and so there will be infinite solutions.

So in most cases elimination fails (e.g. more observations than unknowns, noise in the measurements). In this case we can instead find the 'closest' fit to the data by projecting the outcome vector onto the column space (containing all linear combinations of the predictors), with some remaining error. If there is in fact a solution with 0 error, then we will still find it. However, if there's not, we'll instead return an approximate solution with the minimum possible error for this data set.

#### The normal equation (OLS)

By removing the unsolveable error, we can use the 'normal equation' to solve Ax=p, where A is the feature matrix, x is the coefficients, and p is the solveable part of the outcomes b (p = projection matrix * b). By projecting our outcomes b onto the column space (all linear combinations of the features), we find the part of b that can be explained by our features. By either geometry or algebra we can show the error is actually minimised when it's perpendicular to the column space, giving the least squares solution for the coefficients (the RSS is minimised). 

**Example Projection:** (Strang, G. Introduction to Linear Algebra, pg 221)

<img src="/assets/projection_example.png" alt="projection" width="70%"/>

Given this knowledge, we can flesh out the calculations:

- When the error is multiplied by our matrix A, it equals 0: $$A^T(b-A\hat{x})=0$$. 
- Then we multiply both sides by $$A^T$$ to get a square inevitable matrix: $$A^TA\hat{x}=A^Tb$$. Which rearranges to $$\hat{x}=(A^TA)^\text{-1} A^Tb$$, giving us the x_hat that minimises the error.
- Given the above, we can define p = P * b, where P is the projection matrix $$P=A(A^TA)^\text{-1} A^T$$, and finally $$p = A\hat{x} = A(A^TA)^\text{-1} A^T * b$$.
- Note, if all measurements are perfect, the error=0 (e=b-Ax), as there is an exact solution to Ax=b and the line goes straight through the points (in this case the projection Pb returns b itself as that is the closest). 

For a more detailed refresher on Linear Algebra, Strang's [MIT course](https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/ax-b-and-the-four-subspaces/) is very good. Note, there's no need to scale the features when using the normal equation.

#### Alternative: Gradient Descent

Instead of using the normal equation, for larger samples we could instead find the coefficients that minimise the RSS (or any cost function) with an algorithm called Gradient Descent. The basic idea is to start with some values for the unknown parameters (coefficients), and gradually change them depending on whether the outcome was over or under predicted (using the derivative of the function to determine the direction of the slope). After a number of steps, we hopefully end up at the minimum. We can set the size of each step with the learning rate, but keep in mind too small a step will take a long time while too large a step can cause it to fail to converge (keep overshooting). As the squared error is quadratic, there is only the one global minimum and so Gradient Descent will always converge in the case of linear regression. Note, for multiple features, we repeat this process for each feature. When the range of each feature varies a lot, it will make Gradient Descent slower (descends quicker on small ranges), so it's advised to scale the features first.

### Interpretation

For example, one possible solution might be:

$$\text{predicted sale price in 1000's} = 50 + 30 * \text{nr_bedrooms} + 20 * \text{nr_bathrooms} + 60 * \text{has_pool}$$

In this model, having a pool increases the predicted sale price by 60K, and for each additional bedroom the sale price increases by 30K. 

## Other notes

**Linear regression is a supervised, parametric model**:
- Supervised: For each observation of predictor measurement(s), xi (i=1,..,n), there is an associated response measurement y. Our goal is to use these to predict the response y for unseen observations, or to understand the relationships between x and y in the population.
- Parametric: Makes an assumption about the shape of f (distribution) to simplify the problem. The true f is unlikely to match, causing some under-fitting.

**Regularisation**:
- Lasso regularisation makes a linear model more restrictive (sets a number of them to 0), but could be seen as more interpretable as it leaves a small number of meaningful predictors.

**Generalized Linear Model (GLM)** is a more flexible framework that allows different error distributions and transformations of the outcome (Y). OLS can be thought of as a subset, requiring a normal error distribution. Examples:
- dependent variable is boolean = logistic regression
- dependent variable is an integer count = Poisson regression

Assumptions of Linear Regression
For linear regression to provide reliable results, certain assumptions should hold:

Linearity: The relationship between the dependent and independent variables is linear.
Independence: The residuals (errors) are independent of each other.
Homoscedasticity: The variance of the residuals is constant across all values of the independent variable(s).
Normality: The residuals of the model are approximately normally distributed (for statistical inference purposes).
No multicollinearity: In multiple regression, the independent variables should not be highly correlated with each other.
Evaluating Model Performance
Several metrics are used to evaluate the performance of a linear regression model:

R-squared (R²): This statistic measures how well the independent variables explain the variation in the dependent variable. An R² value close to 1 means the model explains most of the variance.

Mean Squared Error (MSE): This measures the average of the squared differences between the actual and predicted values, giving an indication of the overall error.

Adjusted R-squared: In multiple regression, this metric adjusts the R-squared for the number of predictors, helping to evaluate whether additional predictors improve the model.



Mean squared error (MSE): the average squared difference between the observed and predicted responses (square so negative or under estimation equal to positive or over estimation). Will be small if the predicted and observed are close, and large if for some of the observations they differ substantially (i.e. easily influenced by outliers as we take the average). As the model complexity increases (flexibility increases, degrees of freedom decrease), generally the training MSE will continue to decrease, but the test MSE will eventually start increasing as the model overfits the training data (the supposed patterns  found in the training data don’t exist in the test data, creating a divergence in the train-test performance).




## Alternative: Maximum Likelihood Estimation (MLE)

An alternative method is Maximum Likelihood Estimation (MLE). In a linear model, if the errors belong to a normal distribution the least squares estimators are also the maximum likelihood estimators.


Maximum likelihood estimation can be performed when the distribution of the error terms is known to belong to a certain parametric family ƒθ of probability distributions.[12] When fθ is a normal distribution with zero mean and variance θ, the resulting estimate is identical to the OLS estimate. GLS estimates are maximum likelihood estimates when ε follows a multivariate normal distribution with a known covariance matrix.
Ridge regression[13][14][15] and other forms of penalized estimation, such as Lasso regression,[5] deliberately introduce bias into the estimation of β in order to reduce the variability of the estimate. The resulting estimates generally have lower mean squared error than the OLS estimates, particularly when multicollinearity is present or when overfitting is a problem. They are generally used when the goal is to predict the value of the response variable y for values of the predictors x that have not yet been observed. These methods are not as commonly used when the goal is inference, since it is difficult to account for the bias.
Least absolute deviation (LAD) regression is a robust estimation technique in that it is less sensitive to the presence of outliers than OLS (but is less efficient than OLS when no outliers are present). It is equivalent to maximum likelihood estimation under a Laplace distribution model for ε.[16]

