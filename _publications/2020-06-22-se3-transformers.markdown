---
title: "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks"
date: 2020-06-22
authors: Fabian B Fuchs, Daniel E Worrall, Volker Fischer, Max Welling
venue: NeurIPS
paperurl: https://arxiv.org/abs/2006.10503

---
### Abstract
We introduce the SE(3)-Transformer, a variant of the self-attention module for
3D point clouds, which is equivariant under continuous 3D roto-translations.
Equivariance is important to ensure stable and predictable performance in the
presence of nuisance transformations of the data input. A positive corollary of
equivariance is increased weight-tying within the model, leading to fewer trainable
parameters and thus decreased sample complexity (i.e. we need less training data).
The SE(3)-Transformer leverages the benefits of self-attention to operate on large
point clouds with varying number of points, while guaranteeing SE(3)-equivariance
for robustness. We evaluate our model on a toy N-body particle simulation dataset,
showcasing the robustness of the predictions under rotations of the input. We further
achieve competitive performance on two real-world datasets, ScanObjectNN and
QM9. In all cases, our model outperforms a strong, non-equivariant attention
baseline and an equivariant model without attention.

{% bibliography --query @*[title={{ page.title }}] %}
