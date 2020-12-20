---
title: "Aleatoric and epistemic uncertainties: are they broken?"
date: 2019-12-29 00:00:00 +0000
tags: probability bayesianism Jaynes philosophy
---
Most posts serve to educate by showing how something works, but this one does rather that opposite. It poses a question, one that I have been thinking about for some time. It's about an inconsistent framework for describing aleatoric and epistemic uncertainties.

#### 1 The variance route

Let's begin with a predictive model $$p(y \mid x, \theta)$$, with input $$x$$, real output $$y \in \mathbb{R}$$, and parameters $$\theta$$. Now say that by some process we have trained the model and have a posterior distribution $$p(\theta \mid \mathcal{D})$$ in the parameters given the data. Then we can write the posterior predictive distribution as

$$
p(y \mid x) = \int p(y \mid x, \theta) p(\theta \mid \mathcal{D}) \, \mathrm{d} \theta.
$$

By the [law of total variance](https://en.wikipedia.org/wiki/Law_of_total_variance), we can decompose the variance of the posterior predictive into the following terms

$$
\underbrace{\mathbb{V}_{p(y \mid x)}[y]}_{\text{total variance}} = \underbrace{\color{ForestGreen}{\mathbb{E}_{p(\theta \mid \mathcal{D})}[\mathbb{V}_{p(y \mid x, \theta)}[y]]}}_{\text{(1) aleatoric term}} + \underbrace{\color{MidnightBlue}{\mathbb{V}_{p(\theta \mid \mathcal{D})}[\mathbb{E}_{p(y \mid x, \theta)}[y]]}}_{\text{(2) epistemic term}}
$$

We see that the total variance is the sum of the average predictive variance and the variance of the predictive means. Term (1) is typically said to represent aleatoric uncertainty, being an average over variances it converges to a non-zero value with infinite data. The other term (2) measures fluctuations in the mean prediction across different parameters settings. This goes to zero with infinite data, so is typically associated with epistemic uncertainty. This decomposition is used in papers such as [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) (Kendall and Gal, 2017) and [Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution](https://arxiv.org/abs/1705.00664).

#### 2 The entropic route
In the discrete setting people tend to use a different decomposition, an entropy decomposition. They write the following

$$
  \underbrace{H[Y | X=x]}_{\text{conditional entropy}} = \underbrace{\color{ForestGreen}{H[Y | \Theta, X=x]}}_{\text{(1) aleatoric term}} + \underbrace{\color{MidnightBlue}{\mathbb{I}[Y ; \Theta | X=x]}}_{\text{(2) epistemic term}}.
$$

Here we have paired up the conditional entropy with the total variance. Why does this make sense as a candidate for an aleatoric--epistemic decomposition? Well the aleatoric term (1) shows the residual uncertainty in $$Y$$ given the learned distribution in $$\Theta$$; that is, the variation in $$Y$$ not captured by the model, which is very intuitive. The epistemic term (2) is a little more cryptic. It describes the reduction in uncertainty over the model because of the targets.

The aleatoric term 1) is a mutual information between the parameters and the predictions
