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

For $\textcolor{orange}{\frac{\partial\theta_i}{\partial\mu_i}}$, let's 1st look at the inverse. Remember from previous chapter about the mean and variance of exponential family that $\frac{\partial\mu}{\partial\theta}=V(\mu)$, i.e., that it's equal to the variance function. So, $\frac{\partial\theta_i}{\partial\mu_i}=\frac{1}{V(\mu_i)}$.

For $\textcolor{lime}{\frac{\partial\mu_i}{\partial\eta_i}}$.
- This depends on our specific choice of link function. Since our analysis here is general, we will leave this term as-is for the time being.
- Once we have a link function, this is equal to derivative of the inverse function.
  - I.e., $\mu=g^{-1}(\eta), \frac{\partial\mu}{\partial\eta}=\frac{\partial g^{-1}(\eta)}{\partial\eta}$
  - E.g., suppose we have the logit link function, $g(\mu)=\log(\frac{\mu}{1-\mu}=\eta)$; The inverse of the logit is the sigmoid, $g^{-1}(\eta)=\frac{e^\eta}{1+e^eta}=\sigma(\eta)=\mu$; The derivative of $\mu$ w.r.t. $\eta$ thus becomes $\frac{\partial\sigma(\eta)}{\partial\eta}$, which turns out to be $\sigma(\eta)(1-\sigma(\eta))$

Putting everything together, we get:

$$\frac{\partial\ell_i}{\partial\beta_j}=\frac{y_i-b'(\theta)}{a(\phi)}\cdot\frac{1}{V(\mu_i)}\cdot\frac{\partial\mu_i}{\partial\eta_i}\cdot x_{ij}$$

Remember that $a(\phi)V(\mu_i)=\mathbb{V}[y_i]$ and that $b'(\theta)=\mu$. So we could replace the terms, but it's not essential.

The only thing left to do is to equate this to 0 and solve it! Right? Well … no.

The problem is that this usually doesn't have a close form solution. We will have to use numerical solutions which we will see in the next section.

### :apple: 4.1.2 Canonical link function

If we choose the canonical link function, that is, if we set $\theta=\eta=\beta^Tx$, how will our calculations change?

If $\theta=\eta$, then the middle terms here cancel out:

$$\frac{\partial\ell_i}{\partial\beta_j}=\frac{\partial\ell_i}{\partial\theta_i}\frac{\partial\theta_i}{\partial\mu_i}\frac{\partial\mu_i}{\partial\eta_i}\frac{\partial\eta_i}{\partial\beta_j}=\frac{\partial\ell_i}{\partial\theta_i}\textcolor{orange}{\frac{\partial\theta_i}{\partial\mu_i}\frac{\partial\mu_i}{\partial\theta_i}}\frac{\partial\eta_i}{\partial\beta_j}=\frac{\partial\ell_i}{\partial\theta_i}\frac{\partial\eta_i}{\partial\beta_j}$$

So we would get that:

$$\frac{\partial\ell_i}{\partial\beta_j}=\frac{y_i-b'(\theta)}{a(\phi)}\cdot x_{ij}$$

So we see the equation simplified a lot.

### :apple: 4.1.3 Full derivative and vector notation
This was for a single observation. The full derivative over all our dataset will be a sum over the observations. That is:

$$\frac{\partial\ell_i}{\partial\beta_j}=\sum_{i=1}^n\frac{y_i-b'(\theta)}{a(\phi)}\cdot\frac{1}{V(\mu_i)}\cdot\frac{\partial\mu_i}{\partial\eta_i}\cdot x_{ij}$$

For the canonical link function we will have:

$$\frac{\partial\ell_i}{\partial\beta_j}=\sum_{i=1}^n\frac{y_i-b'(\theta)}{a(\phi)}\cdot x_{ij}$$

To move to vector form and take the derivative w.r.t. the full vector of coefficients $\beta$, the only thing that will be different for the single observation is that $x$ becomes a vector, i.e., we will have:

$$\frac{\partial\ell_i}{\partial\beta_j}=\frac{y_i-b'(\theta)}{a(\phi)}\cdot\frac{1}{V(\mu_i)}\cdot\frac{\partial\mu_i}{\partial\eta_i}\cdot x_i$$

Where $x_i$ is a vector of length $p$ (an intercept and $p-1$ covariates) for the $i$'th observation.

To move to the full data derivative, we can take advantage of matrix operations and write everything in matrix form.
- We will create 2 diagonal $(n\times n)$ matrices:

$$V=diag(V(\mu_1),\cdots,V(\mu_n))=\begin{bmatrix}V(\mu_1)&\dots&0\\ 
\vdots&\ddots&\vdots\\ 
0&\dots&V(\mu_n)\end{bmatrix}$$

$$D=diag(\frac{\partial\mu_1}{\partial\eta_1},\cdots,\frac{\partial\mu_n}{\partial\eta_n})=\begin{bmatrix} \frac{\partial\mu_1}{\partial\eta_1} &\dots&0\\ 
\vdots&\ddots&\vdots\\ 
0&\dots&\frac{\partial\mu_n}{\partial\eta_n}\end{bmatrix}$$

- We will denote $X$ by the design matrix which is of $(n\times n)$ shape
- We will denote $y$ by the $(n\times 1)$ vector of different responses
- We will denote $\mu$ by the $(n\times 1)$ vector of different means

And we will get:

$$\frac{\partial\ell}{\partial\beta}=\frac{1}{a(\phi)}X^TV^{-1}D(y-\mu) \qquad \text{(for column vector)}$$

$$\frac{\partial\ell}{\partial\beta}=\frac{1}{a(\phi)}(y-\mu)^TV^{-1}DX \qquad \text{(for row vector)}$$

For the canonical link function, the middle terms $(V^{-1}D)$ cancel out:

$$\frac{\partial\ell}{\partial\beta}=\frac{1}{a(\phi)}X^T(y-\mu) \qquad \text{(for column vector)}$$

$$\frac{\partial\ell}{\partial\beta}=\frac{1}{a(\phi)}(y-\mu)^TX \qquad \text{(for row vector)}$$

### :apple: 4.1.4 Poisson Example

$$y_i\sim Poisson(\lambda_i)$$

$$f_Y(y_i)=\frac{e^{-\lambda_i}\lambda_i^{y_i}}{y_i!}=\exp[y_i\ln\lambda_i-\lambda_i-\ln y_i!]$$

Mainly for practice, let's remember what are the different terms in the general exponential family representation:
- $a(\phi)=1$
- $\theta_i=\ln\lambda_i\Rightarrow\lambda_i=e^{\theta_i}$
- $b(\theta_i)=e^{\theta_i}\Rightarrow b'(\theta_i)=e^{\theta_i}$
- $c(y_i,\phi)=-\ln y_i!$

Suppose we take the canonical link function, meaning that $\theta_i=\eta_i=\beta^Tx_i$

$$\ell=\sum_{i=1}^n y_i\ln\lambda_i-\lambda_i-\ln y_i!$$

$$\frac{\partial\ell}{\partial\beta_j}=\sum_{i=1}^n\frac{y_i-b'(\theta)}{a(\phi)}\cdot x_{ij}=\sum_{i=1}^n\frac{y_i-e^{\theta_i}}{1}\cdot x_{ij}=\sum_{i=1}^n(y_i-e^{\theta_i})\cdot x_{ij}$$

Or in vector form:

$$\frac{\partial\ell}{\partial\beta_j}=\frac{1}{a(\phi)}X^T(y-\mu)=X^T(y-e^{X\beta})$$

Where $e^{X\beta}$ is taking the element-wise exponent of each element in the $X\beta$ vector.
