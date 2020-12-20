---
title:  "Bayesian Image Quality Transfer with CNNs: Exploring Uncertainty in dMRI Super-Resolution"
date:   2017-09-01
authors: Ryutaro Tanno, Daniel Worrall, Aurobrata Ghosh, Enrico Kaden, Stamatios N Sotiropoulos, Antonio Criminisi, Daniel Alexander
venue: MICCAI
remarks: Winner of Young Scientist Award
paperurl: https://arxiv.org/abs/1705.00664
---
### Abstract

In this work, we investigate the value of uncertainty modelling in 3D
super-resolution with convolutional neural networks (CNNs). Deep learning has
shown success in a plethora of medical image transformation problems, such as
super-resolution (SR) and image synthesis. However, the highly ill-posed nature
of such problems results in inevitable ambiguity in the learning of networks. We
propose to account for intrinsic uncertainty through a per-patch heteroscedastic
noise model and for parameter uncertainty through approximate Bayesian inference
in the form of variational dropout. We show that the combined benefits of both
lead to the state-of-the-art performance SR of diffusion MR brain images in
terms of errors compared to ground truth. We further show that the reduced error
scores produce tangible benefits in downstream tractography. In addition, the
probabilistic nature of the methods naturally confers a mechanism to quantify
uncertainty over the super-resolved output. We demonstrate through experiments
on both healthy and pathological brains the potential utility of such an
uncertainty measure in the risk assessment of the super-resolved images for
subsequent clinical use.

{% bibliography --query @*[title={{ page.title }}] %}
