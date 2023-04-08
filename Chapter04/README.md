# 4 Likelihood Analysis

# :herb: 4.1 Likelihood Analysis

We want to find the coefficients of our model, that is, the $\beta$'s that relate our predictors / covariates to the response:

$$g(\mu)=LP=\eta=\beta^Tx=\beta_0+\beta_1x_1+\cdots$$

We will denote the linear predictor by $\eta$.

Remember that in GLM, we find the predictors that maximize the likelihood. Maximizing the likelihood is exactly like maximizing the log-likelihood (since log is a monotonic transformation). So we will maximize the log-likelihood of our data.

## :apple: 4.1.1 General Analysis
For a given problem we will have to decide
- What is the plausible distribution of our data
- What is the link function that connects the mean of the distribution to the predictors

But for now let's defer the decision, and take a general approach. Taking the most general exponential-family representation of our likelihood, we get:

$$
\begin{aligned}
&f_y(y_i)=\exp(\frac{1}{a(\phi)}(y_i\cdot\theta-b(\theta))+c(y_i,\phi))\\
&\mathcal{L}=\prod_{i=1}^n\exp(\frac{1}{a(\phi)}(y_i\cdot\theta-b(\theta))+c(y_i,\phi))\\
&\ell=\log\mathcal{L}=\log\prod_{i=1}^n\exp(\frac{1}{a(\phi)}(y_i\cdot\theta-b(\theta))+c(y_i,\phi))=\sum_{i=1}^n(\frac{1}{a(\phi)}(y_i\cdot\theta-b(\theta))+c(y_i,\phi))
\end{aligned}
$$

Where are the $\beta$'s here?
- They are hidden in the $\theta$, the natural parameter, since it is a function of the mean $\mu$, which is in itself a function of the $\beta$'s.
  - Remember that $g(\mu)=\eta=\beta^Tx$, so if we take the inverse function we get: $\mu=g^{-1}(\beta^Tx)$, $which is some function of $\beta$ (i.e., we could denote it as $\mu=h(\beta)$)
- So, $\theta(\mu)$, and $\mu(\beta)$, or $\theta(\mu(\beta))$

To do maximum-likelihood we have to differentiate w.r.t. $\beta$. We will use the chain rule here.
- For simplicity we will look at a single observation, $(x_i, y_i)$, which will also mean $\ell_i, \theta_i, \mu_i, \eta_i$.
  - To get the full derivative we just have to take a sum over the observations
- Also we will look only on a single coefficient, e.g., $\beta_j$
  - Later we will write this in vector form

$$\frac{\partial\ell_i}{\partial\beta_j}=\textcolor{magenta}{\frac{\partial\ell_i}{\partial\theta_i}}\textcolor{orange}{\frac{\partial\theta_i}{\partial\mu_i}}\textcolor{lime}{\frac{\partial\mu_i}{\partial\eta_i}}\textcolor{cyan}{\frac{\partial\eta_i}{\partial\beta_j}}$$

Let's break down each component by itself:

$$
\begin{aligned}
&\textcolor{magenta}{\frac{\partial\ell_i}{\partial\theta_i}}=\frac{\partial(\frac{1}{a(\phi)}(y_i\cdot\theta-b(\theta))+c(y_i,\phi))}{\partial\theta_i}=\frac{y_i-b'(\theta)}{a(\phi)} \\
&\textcolor{orange}{\frac{\partial\theta_i}{\partial\mu_i}}=\frac{1}{V(\mu_i)}\\
&\textcolor{lime}{\frac{\partial\mu_i}{\partial\eta_i}}=???\\
&\textcolor{cyan}{\frac{\partial\eta_i}{\partial\beta_j}}=\frac{\partial(\beta^Tx_i)}{\partial\beta_j}=x_{ij}
\end{aligned}
$$
