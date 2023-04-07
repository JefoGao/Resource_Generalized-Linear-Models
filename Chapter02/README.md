# 2 Exponential Family

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

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

So we can write:
- $a(\phi)=1$, no dispersion parameter
- $\theta=\ln(\frac{p}{1-p})$
  - $\Rightarrow p=\sigma(\theta)=\frac{e^\theta}{1+e^\theta}$ 
- $b(\theta)=\ln(\frac{1}{1-p})=\ln(1+e^\theta)$
- $c(\phi,y)=0$

Overall we get:

$$ p(Y=y)=\exp(y\cdot\theta-\ln(1+e^\theta)) $$

### :apple: 2.1.3 Exponential distribution
Exponential distribution $y\sim Exp(\lambda)$

$$f_Y(y)=\lambda e^{-\lambda y} = \exp(-\lambda y + \ln\lambda)$$

So we can write:
- $a(\phi)=1$, no dispersion parameter
- $\theta=-\lambda$
  - $\Rightarrow \lambda=-\theta$ 
- $b(\theta)=-\ln(\lambda)=-\ln(-\theta)$
- $c(\phi,y)=0$

Overall we get:

$$p(Y=y)=\exp(y\cdot\theta+\ln(-\theta))$$

$$
\begin{aligned}
\end{aligned}
$$
### :apple: 2.1.4 Normal distribution
Normal distribution $y\sim N(\mu, \sigma^2)$

$$
\begin{aligned}
f_Y(y) &= \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{1}{2\sigma^2}(y-\mu)^2)\\
&= \exp[-\frac{1}{2\sigma^2}(y^2-2y\mu+\mu^2)-\ln(\sqrt{2\pi}\sigma)]\\
&= \exp[\frac{1}{\sigma^2}(y\cdot\mu-\frac{\mu^2}{2})-\frac{y^2}{2\sigma^2}-\ln(\sqrt{2\pi}\sigma)]
\end{aligned}
$$

So we can write:
- $a(\phi)=\sigma^2$
- $\theta=\mu$
- $b(\theta)=\frac{\mu^2}{2}=\frac{\theta^2}{2}$
- $c(\phi,y)=-\frac{y^2}{2\sigma^2}-\ln(\sqrt{2\pi}\sigma)$

Overall we get:

$$p(Y=y)=\exp(\frac{1}{\sigma^2}(y\cdot\theta-\frac{\theta^2}{2})-\frac{y^2}{2\sigma^2}-\ln(\sqrt{2\pi}\sigma))$$

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 2.2 More Examples

### :apple: 2.2.1 Non exponential family
Before we go on the other distributions, we want to know first what isn't in the exponential family.
- Uniform
- T
- Mixtrue distributions (e.g. GMM's)

### :apple: 2.2.2 Binomial
Let's transform our $y$'s by dividing it by $n$, to get the empirical ratios $y\in[0,\frac{1}{n},\frac{2}{n},\cdots,1]$

$$y^*=n\cdot y\sim Bin(n,p)$$

$$
\begin{aligned}

\end{aligned}
$$
