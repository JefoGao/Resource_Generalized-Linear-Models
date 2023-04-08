# 3 Deviance

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## 3.1 :herb: Unit Deviance

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

## 3.2 Examples of the unit deviance
### 3.2.1 Poisson

$$
\begin{aligned}
p(Y=y)=\exp[y\cdot\ln\lambda - \lambda - \ln y!]
\end{aligned}
$$
