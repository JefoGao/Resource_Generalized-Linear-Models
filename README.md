# 0 GENERALIZED LINEAR MODELS

|Table of Sections|
|--|
|[:herb: 0.1 Regression](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-01-regression)<br>+-- [:apple: 0.1.1 Parametric Regression](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#apple-011-parametric-regression)<br>+-- [:apple: 0.1.2 Non-Parametric Regression](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#apple-012-nonparametric-regression)<br>+-- [:apple: 0.1.3 Why Use Models?](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#apple-013-why-use-models)<br>+----- [:bread: 0.1.3.1 Inference](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#bread-0131-inference)<br>+----- [:bread: 0.1.3.2 Prediction](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#bread-0132-prediction)<br>[:herb: 0.2 Generalized Linear Models](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-02-generalized-linear-models)<br>[:herb: 0.3 Linear Models](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-03-linear-models)<br>[:herb: 0.4 Finding The Coefficients – Least Squares](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-04-finding-the-coefficients--least-squares)<br>[:herb: 0.5 Weights](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-05-weights)<br>[:herb: 0.6 Estimating $\sigma^2$](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-06-estimating-σ2)<br>[:herb: 0.7 Maximum Likelihood](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-07-maximum-likelihood)<br>[:herb: 0.8 Variance Of Fitted Coefficients](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-08-variance-of-fitted-coefficients)<br>[:herb: 0.9 Inference On $beta$](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-09-inference-on-beta)<br>[:herb: 0.10 Uncertainty Of Prediction](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-010-uncertainty-of-prediction)<br>[:herb: 0.11 How Good Is Our Model? R2, F Test And Aic/Bic](https://github.com/bosoninfo/Resource_Generalized-Linear-Models#herb-011-how-good-is-our-model-r2-f-test-and-aicbic)|

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

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

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.2 Generalized Linear Models

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

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

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

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

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

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.5 Weights

We can also incorporate weights into the analysis. There are different reasons why we might choose to give more weight to certain observations over other observations. This in turn will pull the regression line to be closer to the observations with more weights. 

For example, each point might represent multiple observations. So each point $(x_i,y_i)$ has a corresponding $w_i = n_i$ "weight". Another example is heteroskedastic variance: suppose we know some regions of $x$ have lower variance than others – this implies we need to give more weight for observations in the lower variance regions than the ones in the higher regions. And other reasons exist.

If we would have used weights which give more importance to some points (and hence errors) than others, and express them as a diagonal $(n,n)$ weight matrix $W$, the solution to the optimization problem would change to:

$$\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^n w_ie_i^2 = \arg\min_{\beta} (y-X\beta)^T W (y-X\beta)$$

And the solution would become:

$$\hat{\beta} = (X^T W X)^{-1}X^T W y$$

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.6 Estimating $σ^2$

Let’s define the sum of squared errors using the optimal parameters $\hat{\beta}$ as RSS – Residual Sum of Squares:

$$
\text{RSS}=\sum_{i=1}^n(y_i-\hat{\beta}^T x_i)^2=\sum_{i=1}^n(y_i-\hat{\mu_i})^2
$$

where $\hat{\mu_i}$ is defined to be $\hat{\beta}^T x_i$. We are trying to estimate $\mathbb{V}[y_i]$ which by definition of Variance is $=\mathbb{E}[(y_i-μ_i)^2]$, and so it makes sense to estimate it using the data, replacing the theoretical mean with the sample average (“Monte-Carlo estimation”):

$$
\mathbb{V}[y_i]=\mathbb{E}[(y_i-μ_i)^2] \approx \frac{1}{n} \sum_{i=1}^n (y_i-\hat{\mu_i})^2 = \frac{\text{RSS}}{n}
$$

This turns out to be a biased estimator, due to the coefficients estimation already preformed for the $\beta$’s (who are hidden in the $\mu$’s). Instead we will usually use the unbiased estimator: 

$$ s^2=\frac{\text{RSS}}{n-(p+1)} $$ 

By dividing by $n-(p+1)$ instead of $n$, we are accounting for the fact that the estimated coefficients use up $p+1$ degrees of freedom, leaving only $n-(p+1)$ degrees of freedom to estimate $\sigma^2$. The formula also takes into account the fact that the estimator uses information from the data to estimate $\sigma^2$, which can lead to bias if not properly accounted for. The squared-root of it is also called the “Residual Standard Error”.

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.7 Maximum Likelihood
If we assumed the normal distribution on $y$, we could have also used Maximum Likelihood instead of Least-Squares to find $\beta$ and $\sigma$. As shown in the 2nd intro video, in the case of the normal distribution – both methods are equivalent, and actually compute the same thing: 

The MLE for $\beta$ comes out $=(X^TX)^{-1}X^Ty$

The MLE for $\sigma^2$ comes out $= \frac{RSS}{n}$ (the biased estimator we found).

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.8 Variance of Fitted Coefficients
After fitting the beta coefficients using the data, we can check what is the variability of the coefficients:

$$ Var[\hat{\beta}] = Var[(X^TX)^{-1}X^Ty] $$

Note that $(X^TX)^{-1}X^T$ is considered a known constant matrix multiplied by the random vector $y$, as such we will take it as the left side multiplication, and it’s inverse as the right side multiplication:

$$
\begin{aligned}
Var[(X^TX)^{-1}X^Ty] &= (X^TX)^{-1}X^T \cdot Var[y] \cdot ((X^TX)^{-1}X^T)^T \\
&= (X^TX)^{-1}X^T \cdot \sigma^2 I \cdot X(X^TX)^{-1} \\
&= \sigma^2 (X^TX)^{-1} (X^TX (X^TX)^{-1}) \\
&= \sigma^2 (X^TX)^{-1}
\end{aligned}
$$

We have used the homoscedasticity assumption for $Var[y] = \sigma^2\cdot I$ where $I$ is the $(n,n)$ identity matrix. We also used the fact that $(A^{-1})^{T} = (A^T)^{-1}$. We usually also replace $\sigma^2$ with its estimation to get an estimate for the variance of the coefficients.

Once we have the variance of the coefficients, we can build confidence intervals on them or conduct hypothesis testing (either using the normal distributional assumption, or asymptotic theory).

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.9 Inference on $\beta$

When assuming a normal distribution on the response ($y$'s) (or asymptotic theory for large enough $n$), we don’t just get the mean and variance of $\hat{\beta}$, but the whole distribution. Remember that $\hat{\beta} = (X^TX)^{-1}X^Ty$, and since $X$ are fixed, the entire $M:=(X^TX)^{-1}X$ matrix can be thought of as fixed matrix. When multiplying a normal distributed vector with a fixed matrix we get that the result is still normally distributed:

$$
\begin{aligned}
&y \sim N(X\beta,\sigma^2 \cdot I) \\
\hat{\beta} &= My \sim N(MX\beta, \sigma^2 M^TM) \\
&= N((X^TX)^{-1}X^TX \beta, (X^TX)^{-1}X^T \sigma^2 I ((X^TX)^{-1}X^T)T) \\
&= N(\beta, \sigma^2 (X^TX)^{-1}) 
\end{aligned}
$$

If we assume $\sigma$ is known, then (denoting $A=(X^TX)^{-1}$):

$$ \frac{\hat{\beta_j}-\beta_j}{\sqrt{Var(\hat{\beta_j})}} = \frac{\hat{\beta_j}-\beta_j}{ \sigma\sqrt{A_{jj}} } \sim N(0,1) $$

If we use the estimate for $s^2=\hat{\sigma}^2$ and denote the variance of the $j$'th estimate coefficient using this estimation with $\hat{\mathbb{V}}(\hat{\beta_j})=s^2 A_{jj}$ then:

$$ \frac{\hat{\beta_j}-\beta_j}{\sqrt{\hat{\mathbb{V}}(\hat{\beta_j})}} = \frac{\hat{\beta_j}-\beta_j}{s\sqrt{A_{jj}}} \sim t_{n-p-1} $$

Because it can be shown that $\frac{n-p-1}{\sigma^2} s^2 = \frac{RSS}{\sigma^2} \sim \chi_{n-p-1}^2$, and the fact that

$$\frac{\hat{\beta_j}-\beta_j}{s\sqrt{A_{jj}}} = \frac{\frac{\hat{\beta_j}-\beta_j}{s\sqrt{A_{jj}}}}{ \sqrt{ \frac{n-p-1}{\sigma^2}s^2/(n-p-1) } } = \frac{Z}{\sqrt{\frac{\chi_{n-p-1}^2}{n-p-1}}}\sim t_{n-p-1}$$

Where Z is a standard normal distributied statistic, and $\chi^2_{n-p-1}$ is a chi-square with $n-p-1$ distributed statistic.

We can thus develop hypothesis tests and confidence intervals on $\beta$.

For confidence intervals we get:

$$ \hat{\beta_j} \pm Z_{\alpha/2} \sqrt{Var(\hat{\beta_j})} \quad \text{(when } \sigma^2 \text{ is known)} $$

$$ \hat{\beta_j} \pm t_{\alpha/2,n-p-1} \sqrt{\hat{Var}(\hat{\beta_j})} \quad \text{(when } \sigma^2 \text{ is estimated)} $$

where $t_{\alpha/2,n-p-1}$ refers to the $\alpha/2$ quantile of the $t_{n-p-1}$ distribution. 

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.10 Uncertainty of Prediction

We can also get the uncertainty over a prediction. Suppose we have a new observation $x_* $, then the regression line prediction is:

$$\hat{\mu_* } = (x_* )^T \hat{\beta}$$

And the variance for that prediction is:

$$
\begin{aligned}
\mathbb{V}[\hat{\mu_* }] &= \mathbb{V}[(x_* )^T \hat{\beta}]  \\
&= \mathbb{V}[(x_* )^T (X^TX)^{-1} X^Ty] \\
&= (x_* )^T (X^TX)^{-1} X^T \cdot \sigma^2 \cdot I \cdot X (X^TX)^{-1} x_* ] \\
&= (x_* )^T(X^TX)^{-1}x_* \sigma^2
\end{aligned}
$$

Here too, if we use the normality assumption, or asymptotic theory for large enough samples, we get that the distribution on the prediction is a normal distribution. And we can build confidence intervals, with either the normal quantile or the t quantile, depending if we assume $\sigma$ is known or not:

$$\mu^* \pm Z_{\alpha/2} \sqrt{\mathbb{V}[\mu^* ]} \quad \text{(when } \sigma^2 \text{ is known)} $$

$$\mu^* \pm t_{\alpha/2, n-p-1} \sqrt{\mathbb{V}[\mu^* ]} \quad \text{(when } \sigma^2 \text{ is estimated)} $$

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 0.11 How good is our model? R2, F test and AIC/BIC

We would like to devise tools to measure how good is our model, and to compare different models.

The basic way to do so is by analysis of variance. We can decompose our observation $y_i$ as:

$$
y_i = \hat{\mu_i} + (y_i - \hat{\mu_i}) = \hat{\mu_i} + e_i = \text{fit} + \text{residual}
$$

If we “de-mean” the observations (subtract the mean and center them around 0), we would get:

$$
y_i - \bar{y} = \hat{\mu_i} - \bar{y} + (y_i - \hat{\mu_i})
$$

If we now square each side and sum over all observations we would get:

$$
\begin{aligned}
\sum_{i}(y_i - \bar{y})^2 &= \sum_{i}(\hat{\mu_i} - \bar{y} + (y_i - \hat{\mu_i}))^2 \\
&= \sum_{i}(\hat{\mu_i} - \bar{y})^2 + \sum_{i}(y_i - \hat{\mu_i})^2 + 2\sum_{i}(\hat{\mu_i} - \bar{y})(y_i - \hat{\mu_i})
\end{aligned}
$$

Looking at the last term,

$$
\sum_{i}(\hat{\mu_i} - \bar{y})(y_i - \hat{\mu_i}) = \sum_{i}\hat{\mu_i}(y_i - \hat{\mu_i}) - \bar{y}\sum_{i}(y_i - \hat{\mu_i})
$$

The 2nd term is the sum of the residuals which is equal to 0. The 1st term is also equal to 0, which is easier to see in matrix notation:

$$
\begin{aligned}
\sum_{i}\hat{\mu_i}(y_i - \hat{\mu_i}) &= (X\hat{\beta})^T(y - X\hat{\beta}) \\
&= \hat{\beta}^TX^T(y - X\hat{\beta}) \\
&= \hat{\beta}^T[X^T(y - X(X^TX)^{-1}X^Ty)] 
\end{aligned}
$$

Where

$$
\begin{aligned}
X^T(y - X(X^TX)^{-1}X^Ty) &= X^Ty-X^TX(X^TX)^{-1}X^Ty \\
&=X^Ty - X^Ty = 0
\end{aligned}
$$

We get,

$$
\sum_{i}(y_i - \bar{y})^2 = \sum_{i}(\hat{\mu_i} - \bar{y})^2 + \sum_{i}(y_i - \hat{\mu_i})^2
$$

The left-hand-side (LHS) is called the Total Sum of Squares (TSS) – it is the variance of the $y$’s (multiplied by $n$). The 1st term on the right-hand-side (RHS) is called the Explained Sum of Squares, or the Regression Sum of Squares (ESS) – it is the part of the total variance explained by the regression model. The 2nd term on the RHS is the familiar RSS – it is the part of the total variance not explained by the regression model. So we got that,

$$
TSS = ESS + RSS
$$

$R^2$ is defined to be $ESS/TSS = 1-RSS/TSS$ – i.e., the proportion of the data-variance that is explained by the model. It is a number between 0 and 1, where 1 means a perfect fit, and 0 means no fit at all. It can also be used to compare models. Since adding variables to the model can only increase the $R^2$, there’s also an “adjusted $R^2$” which penalizes using too many predictors.

We can devise a statistical test under a null hypothesis that all (or some) of the predictor coefficients are actually equal to 0 (i.e., the predictor doesn’t affect the response, and should not be included in the model). This is the F-test (because the statistic we devise follows the F distribution). For checking the null hypothesis that none of the predictors should be in the model, the statistic is:

$$
F=\frac{ESS}{(p+1)−1}/\frac{RSS}{n−(p+1)} \sim F_{p,n−p−1}
$$

The denominators in each part are the relevant Degrees of Freedom (DF). The RSS divided by its DF is also called Mean Squared Error (MSE). The test of significance is if the probability of getting the value of the statistic or higher is below some p. value or not.

For checking the null hypothesis that some of the predictors shouldn’t be in the model, we essentially compare “nested” models:

- Model A: $\mu=β_0+β_1x_1$
- Model B: $μ=β_0+β_1x_1+β_2x_2$

The statistic here is:

$$
\frac{\frac{RSS_A−RSS_B}{DF_A−DF_B}}{\frac{RSS_B}{DF_B}} \sim F_{DF_A−DF_B,DF_B}
$$

It can be used to select a set of predictors that should be included in the model. Though the order in which we test might give different results of significance. There are different automatic procedures that utilizes this F-test to select the predictors. This is called Stepwise Regression.

If we want to compare non-nested models, then the AIC or BIC can be used: these are measures that calculate some function of the RSS + penalization for the number of predictors/covariates in the model.

- AIC: $nlog(\frac{RSS}{n})+2⋅(p+1)$
- BIC: $nlog(\frac{RSS}{n})+log(n)⋅(p+1)$

BIC usually penalizes more than AIC, and hence will tend to select models with fewer predictors. In both cases, a lower score is better.
