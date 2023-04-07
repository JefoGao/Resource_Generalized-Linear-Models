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
  - $y$ times a parameter $\theta$; is called the "natural" parameter, and is a function of the mean
a function of the natural parameter
we have another component that is a function only of the data and the dispersion parameter, meaning it's not a function of the mean, hence not a function of the regression coefficients (who relate to the mean via the link function )
