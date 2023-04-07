# 2 Exponential Family

## :herb: 2.1 Definition and Example

The exponential family, also known as the Exponential Dispersion Model (EDM) is a general way of writing many of known distributions.

### :apple: 2.1.1 Definition
A distribution is part of the Exponential Family if it's PDF (probability density function)/PMF (probability mass function) can be written in the following form:

$$
f_Y(y) = \exp(\frac{1}{a(\phi)}(y\cdot\theta-b(\theta))+c(y,\phi))=e^{\frac{1}{a(\phi)}(y\cdot\theta-b(\theta))+c(y,\phi)}
$$

So we can write the PDF/PMF as:
- An exponential; in the exponential we have
  - $a(\phi)$ which is a function of a dispersion parameter $\phi$
  - $y$ times a parameter $\theta$; $\theta$ is called the "natural" parameter, and is a function of the mean $\theta(\mu)$
  - a function of the natural parameter $b(\theta)$
  - we have another component that is a function only of the data and the dispersion parameter, meaning it's not a function of the mean, hence not a function of the regression coefficients (who relate to the mean via the link function $g(\mu)=\beta^Tx$)

When trying to estimate the $\beta$'s value, we don't care so much about $\phi$; it only plays a role when estimating the variance of the parameters. But for now we will focus on estimating the $\beta$'s.

### :apple: 2.1.2 Bernoulli distribution
Bernoulli distribution $y\sim Ber(p)$

$$
\begin{aligned}
p(Y=y) &= p^y(1-p)^{1-y}\\
&= \exp(\ln[p^y(1-p)^{1-y}])\\
&= \exp[y\ln p + (1-y)\ln (1-p)]\\
&= \exp[y(\ln p - \ln(1-p))+\ln(1-p)]\\
&= \exp[y\cdot \ln(\frac{p}{1-p})-\ln(\frac{1}{1-p})]
\end{aligned}
$$



$$
\begin{aligned}
\end{aligned}
$$



$$
\begin{aligned}
\end{aligned}
$$



$$
\begin{aligned}
\end{aligned}
$$
