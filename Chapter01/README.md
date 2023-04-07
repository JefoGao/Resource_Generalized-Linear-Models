# 1 LINEAR MODELS VS. GENERALIZED LINEAR MODELS

## :herb: 1.1 Linear Models

3 assumptions:
  1. $y$'s are independent
  2. Each observation comes from a normal distribution: $y_i \sim (\mu, \sigma^2)$
  3. Means $\mu_i$ are related to predictor variables $x_i$ by a linear model: $\mu_i = x_i^T \beta = \beta_0+\beta_1x_1 + \beta_2x_2 \cdots$

- Coefficients are linear
- Predictor variables can be transformed, but coefficients must be linear

## 1.2 Generalized Linear Models:
Takes linear models and generalizes it by:
  1. Allowing $y$'s to come from any exponential family distribution
  2. Allowing means $\mu$ to be related to predictor variables $x_i$ by some function of $\mu$: $g(\mu_i) = x_i^T \beta$

In regular linear model, method for finding coefficients is ordinary least squears (OLS) and maximum likelihood estimation (MLE), they both gives the same results.

In generalized linear model, you can only use maximum likelihood. Ordinary least squares and maximum likelihood can be seen as a special case of the more general generalized linear model or GLM.

|Linear Model|Generalized Linear Model|
|--|--|
|1. iid|1. iid|
|2. $y\sim N(\mu, \sigma^2)$|2. $y\sim Expo. Family$|
|3. $\mu = x^T \beta$|3. $g(\mu) = x^T \beta$|
|4. OLS or MLE|4. MLE|
