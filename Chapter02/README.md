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
p(nY=ny) &= \binom{n}{ny}p^{ny}(1-p)^{n-ny}\\
&= \exp[ny\ln p + (n-ny)\ln(1-p)+\ln\binom{n}{ny}]\\
&= \exp[n(y\cdot\ln(\frac{p}{1-p})+\ln(1-p))+\ln\binom{n}{ny}]
\end{aligned}
$$

So we see we can write:
- $a(\phi)=\frac{1}{n}$
- $\theta=\ln(\frac{p}{1-p})$
  - $\Rightarrow p=\sigma(\theta)=\frac{e^\theta}{1+e^\theta}$ 
- $b(\theta)=\ln(\frac{1}{1-p})=\ln(1+e^\theta)$
- $c(\phi,y)=\ln\binom{n}{ny}$

Overall we can get:

$$p(nY=ny)=\exp[n(y\cdot\theta+\ln(1+e^\theta))+\ln\binom{n}{ny}]$$

### :apple: 2.2.3 Poisson

Possion distribution $y\sim Poisson(\lambda)$

$$
\begin{aligned}
p(Y=y) &= \frac{e^{-\lambda}\lambda^y}{y!}\\
&= \exp[y\cdot\ln\lambda-\lambda-\ln(y!)]
\end{aligned}
$$

So we see we can write:
- $a(\phi)=1$
- $\theta=\ln\lambda$
  - $\Rightarrow \lambda=e^\theta$ 
- $b(\theta)=e^\theta$
- $c(\phi,y)=-\ln(y!)$

Overall we can get:

$$p(Y=y)=\exp[y\cdot\theta-e^\theta-\ln(y!)]$$

### :apple: 2.2.4 Negative Binomial
Negative Binomial distribution $y\sim NB(r,p)$, where $r$ = # of failures before the experiment is stopped (sometimes defined oppositely).

$$p(Y=y)=\binom{y+r-1}{y}(1-p)^rp^y$$

Let's to write this in terms of the mean so we better understand the relations between the mean and the natural parameter (we could have left it as is, but is not the mean):

$$\mu=\mathbb{E}[y]=\frac{pr}{1-p} \Rightarrow p=\frac{\mu}{\mu+r}$$

So we have,

$$
\begin{aligned}
p(Y=y) &= \binom{y+r-1}{y}(1-p)^rp^y\\
&= \binom{y+r-1}{y}(1-\frac{\mu}{\mu+r})^r(\frac{\mu}{\mu+r})^y \\
&= \binom{y+r-1}{y}(\frac{r}{\mu+r})^r(\frac{\mu}{\mu+r})^y\\
&= \exp[y\ln(\frac{\mu}{\mu+r})+r\ln(\frac{r}{\mu+r})+\ln\binom{y+r-1}{y}]
\end{aligned}
$$

So we see we can write:
- $a(\phi)=1$
- $\theta=\ln(\frac{\mu}{\mu+r})=\ln p$
  - $\Rightarrow \frac{\mu}{\mu+r}=e^\theta$ 
  - $\Rightarrow \mu=\mu\cdot e^\theta+r\cdot e^\theta$
  - $\Rightarrow \mu=\frac{r\cdot e^\theta}{1-e^\theta}$
- $b(\theta)=-r\ln(\frac{r}{\mu+r})=r\ln(\frac{\mu+r}{r})=r\ln(\frac{\frac{r}{1-e\theta}}{r})=-r\ln(1-e^\theta)$
- $c(\phi,y)=\ln\binom{y+r-1}{y}$

Overall we can get:

$$p(Y=y)=\exp[y\cdot\theta+r\ln(1-e^\theta)+\ln\binom{y+r-1}{y}]$$

### :apple: 2.2.5 Gamma
Gamma distribution $y\sim Gamma(\alpha,\beta)$

$$f_Y(y)=\frac{1}{\Gamma(\alpha)}\beta^\alpha y^{\alpha-1} e^{-\beta y}$$

Let's to write this in terms of the mean so we better understand the relations between the mean and the natural parameter (we could have left it as is, but is not the mean):

$$\mu=\mathbb{E}[y]=\frac{\alpha}{\beta} \Rightarrow \beta=\frac{\alpha}{\mu}$$

So we have,

$$
\begin{aligned}
p(Y=y) &= \frac{1}{\Gamma(\alpha)}\beta^\alpha y^{\alpha-1} e^{-\beta y}\\
&= \frac{1}{\Gamma(\alpha)}(\frac{\alpha}{\mu})^\alpha y^{\alpha-1} e^{-\frac{\alpha}{\mu} y}\\
&= \exp[-\frac{\alpha}{\mu}y-\alpha\cdot\ln\mu+(\alpha-1)\ln y + \alpha\ln\alpha-\ln\Gamma(\alpha)]\\
&= \exp[\alpha(-\frac{1}{\mu}y-\ln\mu)+(\alpha-1)\ln y + \alpha\ln\alpha-\ln\Gamma(\alpha)]
\end{aligned}
$$

So we see we can write:
- $a(\phi)=\frac{1}{\alpha}$
- $\theta=-\frac{1}{\mu}$
  - $\Rightarrow \mu=-\frac{1}{\theta}$ 
- $b(\theta)=\ln(-\frac{1}{\theta})$
- $c(\phi,y)=(\alpha-1)\ln y + \alpha\ln\alpha-\ln\Gamma(\alpha)$

Overall we can get:

$$p(Y=y)=\exp[\alpha(y\cdot\theta-\ln(-\frac{1}{\theta}))+(\alpha-1)\ln y + \alpha\ln\alpha-\ln\Gamma(\alpha)]$$

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 2.3 Important Property and GLM Exponential Family

### :apple: 2.3.1 Further explanation
We introduced the Exponential-Family in the above sections. In our presentation of it, we described it as a distribution of the form:

$$p(y)=\exp(\frac{1}{a(\phi}[y\cdot\theta-b(\theta)+c(y,\phi)])$$

You can also somtime show it as:

$$p(y)=h(y,\phi)\exp(\frac{1}{a(\phi}[y\cdot\theta-b(\theta)]), \quad \text{where } h(y,\phi) := \exp(c(y,\phi))$$

- $θ$ is the “natural” parameter
- $b(θ)$ is the log-normalizer
- $a(ϕ)$ is a function of a dispersion parameter
- $h$ is sometimes referred to as the base-measure. 

We saw that many known distributions are actually part of the Expo-Family, including: Normal, Binomial, Poisson, Multinomial and Gamma. Sometimes you will see different symbols for the different parts of this representation, and also just $ϕ$ instead of $a(ϕ)$ for the dispersion parameter.

**An important property** not mentioned in the above sections is that for the Expo-Family, the support of the distribution must not depend on the natural parameters. Thus, distributions such as Uniform and Pareto (without known minimum) are not part of the Expo-Family.

### :apple: 2.3.2 Exponential Family vs. GLM Exponential Family
If you have googled Exponential Family, or studied it in other courses, you might encountered other definitions which are similar but not exact. The GLM definition (also known as Exponential Dispersion Models [EDM]) is actually a bit “simplified” in the sense that it’s very “GLM” oriented. So EDM’s (in our case) are a subset of exponential families.

There are a few differences:
- In GLM’s we make an explicit distinction between the dispersion parameter and the natural (mean-oriented) parameter
- In GLM’s there’s only $y$, our response variable of interest, not a sufficient statistic of it $t(y)$ and not a vector of it. In general Expo-Families, there might be a vector of sufficient statistics $t(y)$ and a corresponding vector of natural parameters $θ$.

This is important. While there are many families of distribution that belong to the Exponential-Family, only a subset of them belong to EDM’s, and as such not all can be used in the GLM framework (the math just doesn’t turn out right…).

For example, the Beta distribution belongs to the Exponential family, but is not an EDM, so one cannot do Beta regression with the GLM framework. Nevertheless – Beta regression was created separately quite recently, and it’s derivation is somewhat similar to GLM (see [Ferrari and Cribari-Neto, 2004](https://www.ime.usp.br/~sferrari/beta.pdf) and [Cribari-Neto and Zeileis, 2012](https://cran.r-project.org/web/packages/betareg/vignettes/betareg.pdf) for more details.

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 2.4 Exponential Family - Mean and Variance

Remember the exponential family representation of PDF's/PMF's:

$$f_Y(y)=\exp(\frac{1}{a(\phi}[y\cdot\theta-b(\theta)+c(y,\phi)])$$

Let's look a bit more into the $b(\theta)$ function. The is sort of a normalizing constant of the distribution - the exponent though makes it the normalizer of the `log` distribution, and not the actual distribution. Because of this, it is known by a few names:
- Log-normalizer
- Log-partition
  - partition function = another name for a normalizing function which comes from the field of physics (physics names are often very poor when discussing general concepts in statistics). It is a function because it can still depend on parameters; if they are all known it becomes a constant.
- Cumulant
  - It is similar to the Moment-Generating-Function, only with a log
  - It has some cumulative property (hence the name) which we won't touch upon

### :apple: 2.4.1 Mean

Let's see a bit more what it means to be a log-normalizer. Essentially the log-normalizer is the log of the normalizing constant (in our case, times the scaling function), that is

$$b(\theta)=a(\phi)\cdot\log\int\exp(\frac{1}{a(\phi)}(y\cdot\theta-b(\theta)))dy$$

From the property of PDF, we know

$$1=\int f_Y(y)dy=\int\exp(\frac{1}{a(\phi)}(y\cdot\theta-b(\theta)+c(y,\phi))dy=\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))\exp(-\frac{b(\theta)}{a(\phi)})dy$$

Since the 2nd term doesn't depend on y, we can take it out of the integral:

$$1=\exp(-\frac{b(\theta)}{a(\phi)})\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy$$

So $\exp(-\frac{b(\theta)}{a(\phi)})$ is the normalizing constant of the distribution of the terms in the PDF that depends on $y$. Dividing by this term we get:

$$
\begin{aligned}
&\Rightarrow \exp(\frac{b(\theta)}{a(\phi)}) = \int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&\Rightarrow \frac{b(\theta)}{a(\phi)} = \log\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&\Rightarrow b(\theta) = a(\phi)\log\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy
\end{aligned}
$$

So this $b(\theta)$ function is equal to (the dispersion function $a(\phi)$, times) the log of the normalizing function. Hence the name log-normalizer / log-partition.

The 1st derivative of $b(\theta)$ (w.r.t. with reference to $\theta$) is equal to the mean of the distribution. Let's see this:

$$
\begin{aligned}
b(\theta) &= \frac{db(\theta)}{d\theta} \\
&= a(\phi)\frac{d}{d\theta}\log\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&= a(\phi)\frac{1}{\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy}\int \frac{d}{d\theta}\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&= \textcolor{red}{a(\phi)}\frac{1}{\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy}\int \frac{y}{\textcolor{red}{a(\phi)}}\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&= \frac{1}{\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy}\int y\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy = \* \\
& [\int\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy = e^{\frac{b(\theta)}{a(\phi)}}] \\
\* &= e^{-\frac{b(\theta)}{a(\phi)}} \int y\exp(\frac{y\theta}{a(\phi)}+c(y,\phi))dy \\
&= \int y\exp(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi))dy \\
&= \int y\cdot f_Y(y)dy = \mu
\end{aligned}
$$

### :apple: 2.4.2 Variance

The 2nd derivative defines the variance function $V(\mu)$ - that is, it relates the mean and the variance of every exponential family distribution. If we multiply this by the dispersion parameter we get the actual variance of the distribution.

$$b''(\theta)=\frac{d\mu}{d\theta}:=V(\mu)$$

$$
\begin{aligned}
b''(\theta) &= \int y\cdot\frac{d}{d\theta}\exp(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi))dy \\
&= \int y\exp(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi))(\frac{y-b'(\theta)}{a(\phi)})dy \\
&= \frac{1}{a(\phi)}[\int y^2\exp(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi))-b'(\theta)\int y\exp(\frac{y\theta-b(\theta)}{a(\phi)}+c(y,\phi))] \\
&= \frac{1}{a(\phi)}[\int y^2 f_Y(y)dy - \mu\int y f_Y(y)dy] \\
&= \frac{1}{a(\phi)}[\mathbb{E}[y^2]-\mathbb{E}[y]^2] = \frac{\mathbb{V}[y]}{a(\phi)}
\end{aligned}
$$

- Note that in the normal distribution $V[\mu]=1$, i.e., the mean doesn't affect the variance; but in other distributions in the expo. family, the variance depends on the mean.
- $V[\mu]$ uniquely determines the distribution in the exponential family. So if I give you a $V[\mu]& then you know what the assumed distribution is.
- You can go back to the examples in the previous videos, and calculate the mean and the variance function (parameterized by $\theta$).
