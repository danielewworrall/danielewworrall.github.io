---
title: "Harmonic Networks: Deep Translation and Rotation Equivariance"
authors: Daniel Worrall, Stephan Garbin, Daniyar Turmukhambetov, Gabriel Brostow
date: 2017-06-01
venue: 'CVPR'
paperurl: 'https://arxiv.org/abs/1612.04642'
---

Code: https://github.com/deworrall92/harmonicConvolutions

<iframe width="560" height="315" src="https://www.youtube.com/embed/qoWAFBYOtoU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br>
<br>
### Abstract

Translating or rotating an input image should not affect the results of
many computer vision tasks. Convolutional neural networks (CNNs) are already
translation equivariant: input image translations produce proportionate feature
map translations. This is not the case for rotations. Global rotation
equivariance is typically sought through data augmentation, but patch-wise
equivariance is more difficult. We present Harmonic Networks or H-Nets, a CNN
exhibiting equivariance to patch-wise translation and 360-rotation. We achieve
this by replacing regular CNN filters with circular harmonics, returning a
maximal response and orientation for every receptive field patch. H-Nets use a
rich, parameter-efficient and low computational complexity representation, and
we show that deep feature maps within the network encode complicated rotational
invariants. We demonstrate that our layers are general enough to be used in
conjunction with the latest architectures and techniques, such as deep
supervision and batch normalization. We also achieve state-of-the-art
classification on rotated-MNIST, and competitive results on other benchmark
challenges.

{% bibliography --query @*[title={{ page.title }}] %}
