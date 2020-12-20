---
title: "Supervised Uncertainty Quantification for Segmentation with Multiple Annotations"
date: 2019-07-05
authors: Shi Hu, Daniel E Worrall, Stefan Knegt, Bas Veeling, Henkjan Huisman, Max Welling
venue: MICCAI
paperurl: https://arxiv.org/abs/1907.01949

---
### Abstract
The accurate estimation of predictive uncertainty carries importance in medical scenarios such as lung node segmentation. Unfortunately, most existing works on predictive uncertainty do not return calibrated uncertainty estimates, which could be used in practice. In this work we exploit multi-grader annotation variability as a source of 'groundtruth' aleatoric uncertainty, which can be treated as a target in a supervised learning problem. We combine this groundtruth uncertainty with a Probabilistic U-Net and test on the LIDC-IDRI lung nodule CT dataset and MICCAI2012 prostate MRI dataset. We find that we are able to improve predictive uncertainty estimates. We also find that we can improve sample accuracy and sample diversity.

{% bibliography --query @*[title={{ page.title }}] %}
