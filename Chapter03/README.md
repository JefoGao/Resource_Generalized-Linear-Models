# 3 Deviance

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## :herb: 3.1 Unit Deviance

The deviance is a generalization of residuals in OLS. In regular regression (Linear Model) we use the residuals, but in GLM the deviance replaces some of the functions residuals have in LM. Remember the PDF/PMF of expo. family:

$$f_Y(y)=\exp[\frac{1}{a(\phi)}(\textcolor{orange}{y\cdot\theta-b(\theta)})+c(y,\phi)]$$

We are going to denote $t(y,\mu)=\textcolor{orange}{y\cdot\theta-b(\theta)}$

We want to find the maximum of this function w.r.t $\mu$. If we differentiate $t$ w.r.t $\mu$, we will get (using the chain rule on the 2nd term):

$$\frac{\partial t}{\partial\mu}=y\cdot\frac{\partial\theta}{\partial\mu} - \frac{\partial b(\theta)}{\partial\theta}\cdot\frac{\partial\theta}{\partial\mu}$$

Remember from the last chapter that $\frac{\partial\mu}{\partial\theta}=V(\mu)$, hence $\frac{\partial\theta}{\partial\mu}=\frac{1}{V(\mu)}$.

Finally we get: 

$$\frac{\partial t}{\partial\mu}=\frac{y}{V(\mu)}-\frac{\mu}{V(\mu)}=\frac{y-\mu}{V(\mu)}$$

Equating to 0, we get that $\mu=y$. If we keep to do 2nd derivative (although complicated to compute), it shows that this is indeed a maximum.

So, setting $\mu=y$, we get that $t(y,y)$ is the maximum w.r.t. $\mu$ (for any value of $y$), and if $\mu$ changes, the function value goes down.

***:sparkles: Definition :sparkles:***
The unit deviance is defined to be:

$$d(y,\mu)=2(t(y,y)-t(y,\mu))$$

Since the 1st term is the maximum, this quantity is always non-negative.

So the unit deviance is small when $\mu=y$, and it grows when \mu is far from $y$ - just like regular residuals.

## :herb: 3.2 Examples of the unit deviance
### :apple: 3.2.1 Poisson

$$
\begin{aligned}
&p(Y=y)=\exp[y\cdot\ln\lambda - \lambda - \ln y!] \\
&t(y,\mu) = t(y,\lambda) = [y\cdot\ln\lambda - \lambda - \ln y!] \\
&d(y,\mu) = 2[(y\cdot\ln y - \lambda - \ln y!) - (y\cdot\ln\lambda - \lambda - \ln y!)] = 2[y\cdot\ln\frac{y}{\lambda}-(y-\lambda)]
\end{aligned}
$$

### :apple: 3.2.2 Normal

$$
\begin{aligned}
&f_Y(y)=\exp[\frac{1}{\sigma^2}(y\cdot\mu-\frac{\mu^2}{2})-\frac{y^2}{2\sigma^2}-\ln\sqrt{2\pi}\sigma] \\
&t(y,\mu) = y\cdot\mu-\frac{\mu^2}{2} \\
&d(y,\mu) = 2[(y\cdot y-\frac{y^2}{2}) - (y\cdot\mu-\frac{\mu^2}{2})] = y^2 - 2y\mu+\mu^2=(y-\mu)^2
\end{aligned}
$$

Here it comes out that the unit deviance is equal to the squared residual in the Linear-Model. But for other distributions this is not the case.

## :herb: 3.3 Measure of Similarity

Note that the unit deviance is not symmetric, i.e., it's not a proper distance metric, but it is a measure of similarity/dis-similarity:

|![image](https://user-images.githubusercontent.com/19381768/230702439-59f3b1bf-6ee7-43b9-84e9-eba7c472ab02.png)|
|:--:|
|Unit deviance of different GLMs|

## :herb: 3.4 The distribution of the unit deviance

We can write the exponential family representation in terms of the unit deviance:

$$
\begin{aligned}
f_Y(y) &= \exp(\frac{1}{a(\phi)}t(y,\mu)+c(y,\phi)) \\
&= \exp(-\frac{1}{2a(\phi)}\textcolor{cyan}{(-2)t(y,\mu)}+\frac{1}{2a(\phi)}\textcolor{cyan}{2t(y,y)}-\frac{1}{2a(\phi)}2t(y,y)+c(y,\phi))\\
&= \exp(-\frac{1}{2a(\phi)}\textcolor{cyan}{d(y,\mu)}\textcolor{magenta}{-\frac{1}{2a(\phi)}2t(y,y)+c(y,\phi)}) \\
&= \exp(-\frac{1}{2a(\phi)}d(y,\mu)+\textcolor{magenta}{\tilde{c}(y,\phi)})
\end{aligned}
$$

We can approximate this function using Laplace's method (aka Saddle Point approximation)

$$\exp(-\frac{1}{2a(\phi)}d(y,\mu)+\tilde{c}(y,\phi))\approx\frac{1}{\sqrt{2\pi a(\phi)V(y)}}\exp[-\frac{1}{2a(\phi)}d(y,\mu)]$$

Where $V(y)$ is the variance function $V(\mu)$ evaluated for $\mu=y$.

If we use this approximation, then

$$\frac{d(y,\mu)}{a(\phi)}\sim \chi_1^2$$

Which also means that $\mathbb{E}[d]=a(\phi)$

## :herb: 3.5 Total Deviance

The total deviance is simply the sum of the unit deviance for all the observations in our data:

$$D=\sum_{i=1}^n d(y_i, \mu_i), \qquad \frac{D}{a(\phi)}\sim\chi_n^2$$

$\frac{D}{a(\phi)}$ is also called the scaled total deviance.

Notice that $\frac{D}{a(\phi)}$ is equal to: $2(\ell(y,y)-\ell(y,\mu))$.

That is - it's ($2\times$)the log-likelihood of the *saturated model*, where we take $\mu=y$, minus the non-saturated / constrained model.

This is easy to show with exponential family representation, as

$$
\begin{aligned}
&\ell(y,\mu)=\sum_{i=1}^n\frac{1}{a(\phi)}t(y_i,\mu_i)+c(y_i,\phi)\\
&\ell(y,y)=\sum_{i=1}^n\frac{1}{a(\phi)}t(y_i,y_i)+c(y_i,\phi)\\
&2(\ell(y,y)-\ell(y,\mu)) = sum_{i=1}^n\frac{2}{a(\phi)}[t(y_i,y_i)-t(y_i,\mu_i)] = \frac{D}{a(\phi)}
\end{aligned}
$$

## :herb: 3.6 Null and Residual Deviance

These are the Total Deviance, calculated with regards to different fitted models:
- We fit this model using GLM, and then calculate $D=\Sigma_{i=1}^n d(y_i, \hat{mu_i})$, where $\hat{\mu}$ is the predicted mean value from the model.
- The null deviance is calculated with regards to the null model: $g(\mu)=\beta_0$, that is a model that only includes an intercept.
- The residual deviance is calculated with regards to any model we wish to test, i.e.,
  - $g(\mu)=\beta^Tx$

The residual deviance is analogous to the Residual-Sum-of-Squares (RSS). And it actually is equal to it for the normal distribution (but not in other distributions). So we can now have some measure of the Goodness-of-Fit of our model to the data. The Null/Residual deviance still distribute $\chi^2$.
