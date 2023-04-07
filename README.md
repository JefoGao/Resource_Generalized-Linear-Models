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

## :herb: 0.4 Finding the Coefficients – Least Squares

In Linear Models, we use the method of least squares (which is attributed to Gauss from late 18th / early 19th century): we are trying to minimize the squared “errors” between the $y$'s (actual/observed response) and the $\mu$'s (predicted response by our model):

$$e_i^2 = (y_i - \mu_i)^2$$

Why squared? If we would try to simply minimize the sum of the (regular, non-squared) errors $\Sigma_{i=1}^n e_i$ there would be no solution – since we can always put our regression line/plane higher and higher and essentially get $-\infty$. What we really want is that the distance between the points to the line/plane is minimized. Using absolute value $|e_i|$ is more “difficult” mathematically, so we use the squared errors:

$$\beta^{\mathrm{T}} = \arg\min_{\beta}\sum_{i=1}^n e_i^2 = \arg\min_{\beta} \sum_{i=1}^n (y_i - \beta^{\mathrm{T}}x_i)^2 = \arg\min_{\beta} (y - X\beta)^{\mathrm{T}}(y - X\beta)$$

- Here $X$ is the $(n,p+1)$ “design matrix” whose 1st column is made of 1’s
- $y$ is an $(n,1)$ vector
- $\beta$ is a $(p+1,1)$ vector. 
- For notation simplicity, I will drop the underline from the vectors, and count on the learner to understand from context.

We can solve this optimization problem by simple matrix calculus: taking the derivative w.r.t. $\beta$ and equating it to 0:

$$\nabla_{\beta}(y - X\beta)^{\mathrm{T}}(y - X\beta) = \nabla_{\beta}(y^{\mathrm{T}}y - 2\beta^{\mathrm{T}}X^{\mathrm{T}}y + \beta^{\mathrm{T}}X^{\mathrm{T}}X\beta) = -2X^{\mathrm{T}}y + 2X^{\mathrm{T}}X\beta$$

Here, we have used the fact that $(a^{\mathrm{T}}b)^{\mathrm{T}} = b^{\mathrm{T}}a$ for any vectors $a$ and $b$. Now, to compute the gradient of this scalar function with respect to $\beta$, we can differentiate each term in the expanded expression with respect to $\beta$:

$$
\begin{aligned}
\nabla_{\beta}(y^{\mathrm{T}}y) &= 0 \\
\nabla_{\beta}(-2\beta^{\mathrm{T}}X^{\mathrm{T}}y) &= -2X^{\mathrm{T}}y \\
\nabla_{\beta}(\beta^{\mathrm{T}}X^{\mathrm{T}}X\beta) &= \nabla_{\beta} (X\beta)^{\mathrm{T}}(X\beta) = 2X^{\mathrm{T}}X\beta
\end{aligned}
$$

Equating this to 0 we get that the optimal $\hat{\beta}$:

$$
\begin{aligned}
X^{\mathrm{T}}X \hat{\beta} &= X^{\mathrm{T}}y \\
\hat{\beta} &= (X^{\mathrm{T}}X)^{-1}X^{\mathrm{T}}y
\end{aligned}
$$

This last equation is called the “Normal Equations”. We assume here $X^{\mathrm{T}}X$ is invertible, but it’s almost always the case for a design matrix with more rows (observations) than columns (predictors) [also assuming no perfect linear dependence exists between predictors]. This is the solution to “Ordinary Least Squares” (OLS).

## :herb: 0.5 Weights

We can also incorporate weights into the analysis. There are different reasons why we might choose to give more weight to certain observations over other observations. This in turn will pull the regression line to be closer to the observations with more weights. 

For example, each point might represent multiple observations. So each point $(x_i,y_i)$ has a corresponding $w_i = n_i$ "weight". Another example is heteroskedastic variance: suppose we know some regions of $x$ have lower variance than others – this implies we need to give more weight for observations in the lower variance regions than the ones in the higher regions. And other reasons exist.

If we would have used weights which give more importance to some points (and hence errors) than others, and express them as a diagonal $(n,n)$ weight matrix $W$, the solution to the optimization problem would change to:

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n w_ie_i^2 = \arg\min_{\beta} (y-X\beta)^T W (y-X\beta)$$

And the solution would become:

$$\hat{\beta} = (X^T W X)^{-1}X^T W y$$

## ：herb: 0.6 Estimating $σ^2$

Let’s define the sum of squared errors using the optimal parameters $\betâ$ 
as RSS – Residual Sum of Squares:

$$
\text{RSS}=\sum_{i=1}^n(y_i-\betâ ^Tx_i)^2=\sum_{i=1}^n(y_i-\hat{\mu_i})^2
$$

where $\hat{\mu_i}$ is defined to be $\betâ ^Tx_i$. We are trying to estimate $V[y_i]$ which by definition of Variance is $=E[(y_i-μ_i)^2]$, and so it makes sense to estimate it using the data, replacing the theoretical mean with the sample average (“Monte-Carlo estimation”):

$$
V[y_i]=E[(y_i-μ_i)^2] \approx \frac{1}{n} \sum_{i=1}^n (y_i-\hat{\mu_i})^2 = \frac{\text{RSS}}{n}
$$

This turns out to be a biased estimator, due to the coefficients estimation already preformed for the $\beta$’s (who are hidden in the $\mu$’s). Instead we will usually use the unbiased estimator: $s^2=\frac{\text{RSS}}{n-(p+1)}$. The squared-root of it is also called the “Residual Standard Error”.
