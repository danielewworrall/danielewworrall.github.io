---
title: "Affine Self Convolution"
date: 2019-11-18
authors: Nichita Diaconu\*, Daniel E Worrall\*
venue: Preprint
paperurl: https://arxiv.org/abs/1911.07704

---
### Abstract
Attention mechanisms, and most prominently self-attention, are a powerful building block for processing not only text but also images. These provide a parameter
efficient method for aggregating inputs. We focus on self-attention in vision models, and we combine it with convolution, which as far as we know, are the first to
do. What emerges is a convolution with data dependent filters. We call this an
Affine Self Convolution. While this is applied differently at each spatial location,
we show that it is translation equivariant. We also modify the Squeeze and Excitation variant of attention, extending both variants of attention to the roto-translation
group. We evaluate these new models on CIFAR10 and CIFAR100 and show an
improvement in the number of parameters, while reaching comparable or higher
accuracy at test time against self-trained baselines.

{% bibliography --query @*[title={{ page.title }}] %}
