# 0 GENERALIZED LINEAR MODELS

## :herb: 0.1 Regression

Regression is a way to relate explanatory variables (x's) to some response variable of interest (y). There are two broad types of regression:

### :apple: 0.1.1 Parametric Regression
These models have a "model" with some assumptions about the way in which we believe the data should be structured. This family includes:
- Linear Models (LM)
- Generalized Linear Models (GLM)
- Non-Linear Models

### :apple: 0.1.2 Non-Parametric Regression
These models don't make strong assumptions and look at the data for answers. This family includes tools such as Gaussian-Process regression, (K-)Nearest-Neighbors (KNN), Smoothing Splines, Kernel Regressions, etc. 

### :apple: 0.1.3 Why Use Models?
Models are used to better understand reality. In statistical regression models this general/amorphic goal is usually divided into 2 specific/practical goals:

#### :bread: 0.1.3.1 Inference
Inference is used to understand how a predictor `$x$` influences an outcome/response `$y$`. Inference helps to answer questions like:

- What is the direction of the influence – positive or negative?
- Does `$y$` increase with `$x$`? Does it decrease?
- Is the change linear, exponential, logarithmic?

#### :bread: 0.1.3.2 Prediction
Prediction is used to predict `$y$` for a new `$x$`, and to do so well enough. Here we don’t care so much about interpretation – just use whatever you have that gives good predictions (not just on the current data, but to future data as well). The model should generalize well and not over-fit the data.

## :herb: 0.2 GLMs

- GLMs are an extension to Linear Models (LMs).
- Used when LMs are not adequate.
- Suitable for response variable (y) which is binary/proportion, count, or non-negative quantity.
- More appropriate models include:
  - Logistic regression
  - Poisson regression
  - Gamma regression
- GLMs offer a unified framework to conduct regression analysis.
- GLMs work on any distribution from the Exponential Family.
- Available software packages use the IRLS method to fit GLMs numerically.
- R implementation is limited to commonly used distributions, but external packages offer additional GLMs.

## :herb: 0.3 Linear Models

Linear models or linear regression is a statistical model of the form:

$$y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip} + \epsilon_i$$

where $p$ is the number of predictors we have, and this implies $p+1$ unknown coefficients with the intercept.

The model consists of two parts:
- Systematic component (Linear Predictor): $\beta_0 + \beta_1 x_{i1} + \cdots + \beta_p x_{ip}$
- Random component: $\epsilon_i$

The linearity comes from the linearity of the coefficients $\beta$. The predictor terms $x$ are considered known, and from the linearity perspective, we don’t care which transformation is used upon them. 

Linear models assume that:
- $\mathbb{E}[\epsilon_i] = 0$ and $\operatorname{Var}[\epsilon_i] = \sigma^2$ for all $i$.
- The response variable $y$ has mean $\mu_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_dx_{ip}$ and variance $\sigma^2$.
- The observations are independent and identically distributed (IID).

A common assumption is that $\epsilon_i \sim N(0, \sigma^2)$. This is sometimes referred to as "Normal Linear Models", but it is sometimes used also to describe Linear Models. Another fundamental assumption is homoscedasticity, that is, the variance doesn’t change as a function of the mean.

So there are a few major assumptions to conduct a LM analysis:
1. $y$ is IID.
2. Homoscedasticity: $\operatorname{Var}[y_i] = \sigma^2$.
3. The relationship between $y$ and $x$ is linear: $\mu_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_dx_{ip}$.

For assumption 2 to be stronger: not only constant variance, but also the shape of the distribution is Normal

$$ y_i \sim N(\mathbb{E}[y_i],\mathbb{V}[y_i])=N(\mu_i, \sigma^2) $$

