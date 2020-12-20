---
title: "Aleatoric and epistemic uncertainties: a deceptively simple distinction?"
date: 2019-12-29 00:00:00 +0000
tags: probability bayesianism Jaynes philosophy
---
*This is a follow-on post from ["On the 'invention' of randomness"]({% post_url blog/2019-12-15-randomness %})**

# Noise is everything you do not know
Rich Turner.

Previously, I wrote about the apparent invention of intrinsic randomness; this idea that randomness (colloquially referred to as noise) can only ever be perceived and that it fundamentally may not exist. What does exist is *observer ignorance* and sometimes we make the very human mistake of believing that our ignorance about the outside world must in fact be 'real', what Jaynes called a *mind projection fallacy*. At this point, this is a very cerebral, dare I say, academic distinction. It would be nice to see whether this distinction has any real downstream consequences for the likes of real-world scientists other than particle physicists.

A problem of much practical significance---and one that I have devoted some time to---is the distinction between *aleatoric* and *epistemic* uncertainties. This distinction has been acknowledged for quite some time but only received a lot of attention within the deep learning community a few years ago with the release of the paper [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977) (Kendall and Gal, 2017). At pretty much exactly the same time, [Ryutaro Tanno](https://rt416.github.io/) and I had released a version of this for the medical imaging community [Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution](https://arxiv.org/abs/1705.00664). Both papers were flawed, this is why.

# What are aleatoric and epistemic uncertainties
_Aleatoric_ uncertainty describes the statistical aberrations in the consistency of observations. Say you run an experiment $$E$$ multiple times with the same settings and each time it returns different observations $$X_1, X_2, ...$$, that statistical dance about the mean can be regarded as the aleatoric _noise_. If you have real-valued observations, then as simple proxy for aleatoric uncertainty might be the observation variance. No matter how many times you run the experiment, this noise is always present. As such, it is sometimes referred to as _irreducible uncertainty_.

On the other hand _epistemic_ uncertainty is a kind of model-based uncertainty. Say we fit a model $$m$$ to data, what we are really doing is choosing a model $$m$$ from a set of models $$\mathcal{M}$$. In most real-world cases we can never know if the model is the correct one, without an infeasibly large amount of data. So epistemic uncertainty is the uncertainty we have in our choice of model because we just didn't collect enough data. For a continuum of models, we can again use variance as a proxy for epistemic uncertainty.

Aleatoric uncertainty is important because it tells us how consistent our data is and epistemic uncertainty is useful because it tells us whether we can have confidence in our model or need more data. This should help us resolve the age0old question, "Why is my model so crap? Is it the model or the data"? The thing is, I think aleatoric and epistemic uncertainty are actually modelling the same thing...(sort of)!

# Two sides of the same coin?


Let's use an example to illustrate these two ideas.

Consider a generative process. We

- aleatoric is model mismatch
- epistemic is lack of data
- We have `invented` randomness because we are too lazy to model reality
- bias-variance tradeoffs
- Bayes'
- discrete and continuous linear models? decompositions?
-no definitions
