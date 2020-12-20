---
title: "Reversible GANs for Memory-efficient Chest CT Super-resolution and Domain-adaptation in 3D"
date: 2019-05-07
authors: Tycho van der Ouderaa, Daniel E Worrall
paperurl: https://openreview.net/pdf?id=SkxueFsiFV
venue: MIDL
---
### Abstract
Medical imaging data are typically large in size. As a result, it can be difficult to train deep neural models on them. The activations of invertible neural networks do not have to be stored to perform backpropagation, therefore such networks can be used to save memory when handling large data volumes. We use a technique called additive coupling to obtain a memory-efficient partially-reversible image-to-image translation model. With this model, we perform a 3D super-resolution and 3D domain-adaptation task, on both paired and unpaired CT scan data. Additionally, we demonstrate experimentally that the model requires significantly less GPU memory than a model without reversibility.

{% bibliography --query @*[title={{ page.title }}] %}
