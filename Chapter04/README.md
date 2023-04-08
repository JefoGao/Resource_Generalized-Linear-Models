# 4 Likelihood Analysis

|Table of Sections|
|--|
|[:herb: 4.1 Likelihood Analysis](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#herb-41-likelihood-analysis)<br>+-- [:apple: 4.1.1 General Analysis](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-411-general-analysis)<br>+-- [:apple: 4.1.2 Canonical Link Function](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-412-canonical-link-function)<br>+-- [:apple: 4.1.3 Full Derivative And Vector Notation](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-413-full-derivative-and-vector-notation)<br>+-- [:apple: 4.1.4 Poisson Example](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-414-poisson-example)<br>[:herb: 4.2 Newton Raphson](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#herb-42-newton-raphson)<br>[:herb: 4.3 Fisher Scoring](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#herb-43-fisher-scoring)<br>+-- [:apple: 4.3.1 The Fisher Information Matrix](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-431-the-fisher-information-matrix)<br>+-- [:apple: 4.3.2 Fisher Scoring Algorithm](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-432-fisher-scoring-algorithm)<br>+-- [:apple: 4.3.3 Advantages And Disadvantages](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-433-advantages-and-disadvantages)<br>[:herb: 4.4 Iterative Reweighted Least Squares](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#herb-44-iterative-reweighted-least-squares)<br>+-- [:apple: 4.4.1 Algorithm](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-441-algorithm)<br>+-- [:apple: 4.4.2 Intuition](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-442-intuition)<br>+-- [:apple: 4.4.3 Advantages And Disadvantages](https://github.com/bosoninfo/Resource_Generalized-Linear-Models/blob/main/Chapter04/README.md#apple-443-advantages-and-disadvantages)|

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 4.1 Likelihood Analysis

We want to find the coefficients of our model, that is, the $\beta$'s that relate our predictors / covariates to the response:

$$g(\mu)=LP=\eta=\beta^Tx=\beta_0+\beta_1x_1+\cdots$$

We will denote the linear predictor by $\eta$.

Remember that in GLM, we find the predictors that maximize the likelihood. Maximizing the likelihood is exactly like maximizing the log-likelihood (since log is a monotonic transformation). So we will maximize the log-likelihood of our data.

### :apple: 4.1.1 General Analysis
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

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 4.2 Newton Raphson

Newton Raphson is an iterative algorithm for finding the maximum likelihood estimation of the parameters in a generalized linear model (GLM). The basic idea is to start with some initial guess for the parameters and then iteratively update the estimates until convergence.

The algorithm works as follows:

1. Choose some initial values for the parameters, denoted by $\beta^{(0)}$.
2. For each observation $i$ in the data set, compute the predicted mean $\mu_i$ using the current parameter estimates: $\mu_i = g^{-1}(x_i^T\beta^{(t)})$, where $g$ is the link function.
3. Compute the working response $z_i$ for each observation: $z_i = x_i^T\beta^{(t)} + (y_i - \mu_i) g'(\mu_i)$.
4. Compute the working weights $w_i$ for each observation: $w_i = \frac{1}{\text{var}(\mu_i)}$ where $\text{var}(\mu_i)$ is the variance function corresponding to the distribution in the GLM.
5. Update the parameter estimates using the formula: $\beta^{(t+1)} = (\mathbf{X}^TW^{(t)}\mathbf{X})^{-1}\mathbf{X}^TW^{(t)}\mathbf{z}^{(t)}$, where $\mathbf{X}$ is the design matrix, $\mathbf{W}$ is the diagonal matrix of working weights, and $\mathbf{z}^{(t)}$ is the vector of working responses computed in step 3.
6. Repeat steps 2-5 until convergence, which can be determined by checking if the change in the parameter estimates between iterations is small.

One advantage of the Newton Raphson algorithm is that it converges faster than other optimization algorithms, such as gradient descent. However, it can be sensitive to the initial values of the parameters and may not converge if the starting values are too far from the true parameter values.

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 4.3 Fisher Scoring

Fisher Scoring is another iterative algorithm used in generalized linear models (GLMs) to estimate the maximum likelihood parameters. It is similar to the Newton-Raphson method, but instead of using the Hessian matrix to update the parameters, it uses an approximation of the Hessian matrix, known as the Fisher information matrix.

### :apple: 4.3.1 The Fisher information matrix

The Fisher information matrix (FIM) is a matrix of second derivatives of the log-likelihood function with respect to the model parameters. In a GLM, the FIM is defined as:

$$\mathbf{F}(\boldsymbol{\theta}) = -\mathbb{E} \left[ \frac{\partial^2 \ell(\boldsymbol{\theta};\mathbf{y})}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T} \right]$$

where $\boldsymbol{\theta}$ is the vector of model parameters, $\ell(\boldsymbol{\theta};\mathbf{y})$ is the log-likelihood function, and $\mathbb{E}$ denotes the expected value operator.

### :apple: 4.3.2 Fisher Scoring algorithm

The Fisher Scoring algorithm iteratively updates the parameter estimates as follows:

1. Initialize the parameter vector $\boldsymbol{\theta}$.
2. Calculate the score vector $\mathbf{U}(\boldsymbol{\theta})$, which is the gradient of the log-likelihood function with respect to $\boldsymbol{\theta}$.
3. Calculate the Fisher information matrix $\mathbf{F}(\boldsymbol{\theta})$.
4. Update the parameter vector:

$$\boldsymbol{\theta}_{new} = \boldsymbol{\theta}_{old} + \left[ \mathbf{F}(\boldsymbol{\theta}_{old}) \right]^{-1} \mathbf{U}(\boldsymbol{\theta}_{old})$$

5. Repeat steps 2-4 until convergence.

The Fisher Scoring algorithm is more computationally efficient than the Newton-Raphson method because it avoids calculating the Hessian matrix directly, which can be expensive for large datasets.

### :apple: 4.3.3 Advantages and disadvantages

The Fisher Scoring algorithm has several advantages over the Newton-Raphson method:

- It is more computationally efficient because it avoids calculating the Hessian matrix directly.
- It can handle singular or nearly singular Hessian matrices, which can cause problems for the Newton-Raphson method.

However, it also has some disadvantages:

- It can be less stable than the Newton-Raphson method for certain types of models.
- It requires calculating the expected value of the second derivative of the log-likelihood function, which can be difficult or impossible to do analytically for some models.

Overall, the Fisher Scoring algorithm is a useful alternative to the Newton-Raphson method for estimating maximum likelihood parameters in GLMs. It is particularly useful for large datasets or models with singular or nearly singular Hessian matrices.

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 4.4 Iterative Reweighted Least Squares

Iterative Reweighted Least Squares (IRLS) is another algorithm used to estimate the maximum likelihood estimates in generalized linear models (GLMs). IRLS is a gradient descent algorithm that iteratively computes the weights for each observation and updates the estimates of the coefficients until convergence is achieved.

### :apple: 4.4.1 Algorithm

The steps for IRLS are as follows:

1. Initialize the coefficients $\beta$ (e.g., using maximum likelihood estimates from the saturated model or the method of moments)
2. Compute the fitted values $\hat{y}$ using the current coefficients $\beta$
3. Compute the working response $z$ as $z = X\beta + \frac{y-\hat{y}}{g'(\hat{y})}$ where $g'$ is the first derivative of the link function
4. Compute the weights $w$ as $w = \frac{1}{\text{var}(y_i)} = \frac{1}{\text{var}(g^{-1}(\mu_i))} = \frac{1}{g''(\mu_i) \cdot V(\mu_i)}$ where $V(\mu_i)$ is the variance function of the distribution and $g''$ is the second derivative of the link function
5. Compute the weighted least squares estimate of the coefficients $\beta$ as $\beta_{new} = (X^TWX)^{-1}X^TWz$
6. If the difference between the current and updated coefficients is smaller than a specified tolerance, stop. Otherwise, go back to step 2.

### :apple: 4.4.2 Intuition

The intuition behind IRLS is that it is a weighted least squares algorithm, where the weights are chosen to reflect the variance of each observation. In particular, observations with smaller variance are given more weight in the computation of the updated estimates of the coefficients. This means that IRLS gives more weight to observations that are more informative, and less weight to observations that are noisy or less informative.

The working response $z$ is used to adjust for the discrepancy between the observed response $y$ and the fitted values $\hat{y}$, and to weight the observations according to their variance. The weights are used to down-weight observations with high variance, which can cause the estimates of the coefficients to be biased.

### :apple: 4.4.3 Advantages and Disadvantages

The advantages of IRLS are:

- It can handle a wide range of distribution families, including non-normal distributions
- It is computationally efficient and scales well to large datasets
- It can handle missing data by using only the available data to update the estimates of the coefficients

The disadvantages of IRLS are:

- It can be sensitive to the initial values of the coefficients, and may converge to local optima
- It may not converge for some datasets, particularly those with a large number of outliers or influential observations
- It can be difficult to diagnose convergence problems or assess the quality of the estimates

In summary, IRLS is a popular algorithm for estimating the maximum likelihood estimates in GLMs, particularly for models with non-normal distributions or missing data. It is a computationally efficient algorithm that iteratively updates the estimates of the coefficients using a weighted least squares approach. However, it can be sensitive to the initial values of the coefficients and may not converge for some datasets.
