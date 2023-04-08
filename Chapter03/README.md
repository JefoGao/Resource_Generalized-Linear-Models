# 3 Deviance

<p align="center"><img src="https://user-images.githubusercontent.com/19381768/230614263-feff794c-64ca-404b-9e44-849eaebc22fd.png" width=50%/></p>

## 3.1 :herb: Unit Deviance

The deviance is a generalization of residuals in OLS. In regular regression (Linear Model) we use the residuals, but in GLM the deviance replaces some of the functions residuals have in LM. Remember the PDF/PMF of expo. family:

$$f_Y(y)=\exp[\frac{1}{a(\phi)}(\textcolor{orange}{y\cdot\theta-b(\theta)})+c(y,\phi)]$$

We are going to denote $t(y,\mu)=\textcolor{orange}{y\cdot\theta-b(\theta)}$

We want to find the maximum of this function w.r.t $\mu$. If we differentiate $t$ w.r.t $\mu$, we will get (using the chain rule on the 2nd term):

$$\frac{\partial t}{\partial\mu}=y\cdot\frac{\partial\theta}{\partial\mu}-\frac{\partial b(\theta)}{\partial\theta}\cdot\frac{\partial\theta}{\partial\mu}$$
