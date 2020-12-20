---
title: "MDP Homomorphic Networks: Group Symmetries in Reinforcement Learning"
date: 2020-06-30
authors: Elise van der Pol, Daniel E Worrall, Herke van Hoof, Frans A Oliehoek, Max Welling
venue: NeurIPS
paperurl: https://arxiv.org/abs/2006.16908
excerpt: Equivariance + RL
---
### Abstract
This paper introduces MDP homomorphic networks for deep reinforcement learning. MDP homomorphic networks are neural networks that are equivariant under symmetries in the joint state-action space of an MDP. Current approaches to deep reinforcement learning do not usually exploit knowledge about such structure. By building this prior knowledge into policy and value networks using an equivariance constraint, we can reduce the size of the solution space. We specifically focus on group-structured symmetries (invertible transformations). Additionally, we introduce an easy method for constructing equivariant network layers numerically, so the system designer need not solve the constraints by hand, as is typically done. We construct MDP homomorphic MLPs and CNNs that are equivariant under either a group of reflections or rotations. We show that such networks converge faster than unstructured baselines on CartPole, a grid world and Pong.

{% bibliography --query @*[title={{ page.title }}] %}
