# 1 LINEAR MODELS VS. GENERALIZED LINEAR MODELS

## 1.1 Linear Models

3 assumptions:
  1. $y$'s are independent
  2. Each observation comes from a normal distribution: $y_i \sim (\mu, \sigma^2)$
  3. Means $\mu_i$ are related to predictor variables $x_i$ by a linear model: $\mu_i = x_i^T \beta$

Coefficients are linear
Predictor variables can be transformed, but coefficients must be linear

## 1.2 Generalized Linear Models:
Takes linear models and generalizes it by:
  1. Allowing Y's to come from any exponential family distribution
  2. Allowing means mu to be related to predictor variables XI by some function of mu
Method for finding coefficients is only maximum likelihood
Ordinary least squares and maximum likelihood can be seen as a special case of the more general generalized linear model or GLM

Next video will go into more detail on the differences between ordinary least squares and maximum likelihood
