# 1 Introduction to Generalized Linear Models

## :herb: 1.1 Linear Models vs. Generalized Linear Models
### :apple: 1.1.1 Linear Models
3 assumptions:
  1. $y$'s are independent
  2. Each observation comes from a normal distribution: $y_i \sim (\mu, \sigma^2)$
  3. Means $\mu_i$ are related to predictor variables $x_i$ by a linear model: $\mu_i = x_i^T \beta = \beta_0+\beta_1x_1 + \beta_2x_2 \cdots$

- Coefficients are linear
- Predictor variables can be transformed, but coefficients must be linear

### :apple: 1.1.2 Generalized Linear Models:
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

## :herb: 1.2 Least Squares vs. Maximum Likelihood

### :apple: 1.2.1 Least Squares
Least squares is typically portrayed as an optimization problem where a line is fit to lots of points in two or more dimensions. The best line is one where the squared distance between the points and the line is minimized. The problem can be generalized to multiple predictor variables where the goal is to find some linear combination that is as close as possible to the Y values. This problem is solved by differentiating and equating to zero.

- In Least-Squares one tries to minimize the square distance between the observed points and the points predicted by the model (e.g. $\hat{y} = \beta_0+\beta_1x$)
- The minimization is done with regards to (w.r.t. = with reference to) the parameters of the model, i.e. to the $\beta$'s: $\beta_0,\beta_1$ (in 2d example) - which can also be written in vector form simply as $\beta$.
  - The move to vector notation allows us to move to higher dimension. Instead of a line we will fit a plane or a hyperplane.

$$
\min \sum^n_{i=1} (y_i-\hat{y_i})^2 = \min_{\beta_0, \beta_1} \sum^n_{i=1} (y_i-[\beta_0+\beta_1x_i])^2 = \min_{\beta} \sum^n_{i=1} (y_i-\beta^Tx_i])^2
$$

- By differentiating and equating to zero, we can find the best parameters.
  - In the linear case (linear in the parameters, that is linear in $\beta$'s: $\beta^Tx$), we can find a closed form solution.

### :apple: 1.2.2 Maximum Likelihood
Maximum likelihood assumes a distribution on the Y values. In the case of a normal distribution, the mean is assumed to be at the center of the distribution with some standard deviation. The goal is to choose beta coefficients that maximize the likelihood function, which is the product of the probability density function of the normal distribution for each observation. This function is maximized by differentiating the log of the likelihood function and equating to zero.

In the case of a normal distribution, the results of the maximum likelihood method and least squares method are the same. However, if the Y values come from a non-normal distribution such as a Bernoulli distribution or a Poisson distribution, the maximum likelihood method is used to compute the values of the coefficients that will maximize the distribution. In generalized linear models, the Y values are not necessarily from a normal distribution and can come from skewed distributions like a gamma distribution or a chi-squared distribution. 

|![image](https://user-images.githubusercontent.com/19381768/230558868-e7ed823a-caf1-42d7-bb7f-57f8a43d4840.png)|
|:--:|
|Maximum Likelihood Estimation|

In maximum likelihood we assume that the $y$'s distribute, with a mean that depends on $x$;  that means, e.g. for the $y\sim N(\beta^Tx, \sigma^2) case, that for each $x$, we have a normal Gaussian centered around some point. 
- If we connect the centers we would get a straight line
- The means (centers) are different depending on the value of $x$. In the graph, for the lower $x$, the means are lower than for the higher $x$.

How do you compute the maximum likelihood?
- For each $y_i$ I have some probability of obtaining it
  - E.g. in the normal case: $\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2}$
- Let's take the product across all observations; Continuing with the normal case:

$$
\mathcal{L} = \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2}
$$

- The $arg max$ is the same for the likelihood or the log-likelihood, so easier to take the log and turn the product into a sum; Continuing:

$$
\begin{aligned}
\ell := log\mathcal{L} &= log \prod_{i=1}^n \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2}\\
&= \sum_{i=1}^n log (\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2})\\
&= \sum_{i=1}^n [log (\frac{1}{\sqrt{2\pi}\sigma}) - \frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2]
\end{aligned}
$$
  
  - The first term $log (\frac{1}{\sqrt{2\pi}\sigma})$ doesn't depend on the parameters we optimize, so we can discard it

$$
\begin{aligned}
arg \max_{\beta} \ell &= arg \max_{\beta} - \sum_{i=1}^n [\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2] \\
&= arg \min_{\beta} \sum_{i=1}^n [\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2]
\end{aligned}
$$

  - Maximizing a negative quantity is the same as minimizing the positive quantity

$$
arg \min_{\beta} \sum_{i=1}^n [\frac{1}{2\sigma^2}(y_i - \beta^T x_i)^2] = arg min_{\beta} \sum_{i=1}^n (y_i - \beta^Tx_i)^2
$$

  - Again, we see a multiplying constant $\frac{1}{2\sigma^2}$ that doesn't affect the arg min.

We see that for the normal distribution, the least squares and the maximum likelihood methods are actually equivalent - we are doing exactly the same.

BUT - if we assume a different distribution for the $y$ - e.g., Bernoulli, Poisson, etc., the methods won't be equivalent - and we will only use Maximum Likelihood. 

## :herb: 1.3 Saturated vs. Constrained Models

In this section, we gain some more intuition about maximum likelihood by contrasting two possible models: the unconstrained or saturated model and the constraint or regular model. Both models assume the same assumptions as mentioned in a previous video, which are that the observations, after some random response variable, are independent, that they come from some exponential family distribution, and that there is a relation between the means of the distribution and some predictor variable X.

The regular model tries to fit some linear line or other model to the points, and it does it by the method of maximum likelihood. Under the assumption that the mean is the maximum likelihood of the normal distribution, we get our betas that maximize the overall likelihood of all those means.

The saturated model, on the other hand, focuses all the means exactly on the points, so this is the maximum likelihood that is ever possible. You can think of it as if we are fitting some high order polynomial axes to the data. 

In the context of GLM, the method of maximum likelihood finds the maximum likelihood that is possible under some constrained model. We try to find a linear line that maximizes these values that are as close as possible to the center to the maximum likelihood of each observation. The difference between the log likelihood of the saturated model and the regular model is a quantity called a unit deviance, which we will talk about in a future video.

One possible question is if there is a distribution that is not normal or asymmetric, where should we put the mean of this distribution? Should we put the mode of the distribution on this point instead of the mean? The way GLM works is that you always put the mean, and we will also see in the upcoming video is that the math works out for the mean.

This gives us more intuition as to what maximum likelihood is doing in the GLM context.
