---
title: "On rooted trees and differentiation"
date: 2023-11-22                    
permalink: /blog/2023/11/rooted-trees-and-differentiation/
tags:
  - differentiation
  - algebra
excerpt: "The chain rule for higher order derivatives boosts a wealth of beautiful mathematical structure touching the theory of special rooted trees, group theory, combinatorics of integer partitions, order theory, and many others."
---
# Introduction
The chain rule lies at the heart of the backpropagation algorithm in deep learning. Unbeknownst to many though, the chain rule for higher order derivatives boasts a wealth of beautiful mathematical structure touching the theory of special rooted trees, group theory, combinatorics of integer partitions, order theory, and many others. I've been meaning to write this post for a long time, but in the last year work has been quite busy. I'm glad I can finally share with you the beautiful maths connecting special rooted trees and differentiation.


### The chain rule
We start with a composition of functions
\begin{align}
    \textbf{z} = f(g(\textbf{x}))
\end{align}
where $f$ and $g$ are vector-in vector-out functions. We can introduce an intermediate variable $\textbf{y} = g(\textbf{x})$ so that $\textbf{z} = f(\textbf{y})$. The derivative of $\textbf{z}$ with respect to $\textbf{x}$ is then
\begin{align}
    \frac{\partial \textbf{z}}{\partial \textbf{x}} = \frac{\partial \textbf{z}}{\partial \textbf{y}} \frac{\partial \textbf{y}}{\partial \textbf{x}}.
\end{align}
In any contemporary machine learning masters course, this is about as far as we go. Couple the chain rule with dynamic programming and you get the backpropagation algorithm and forward-mode differentiation. And for most practitioners, we do not even need to know as much. With the advent of packages like [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) all this machinery is hidden away. Well not today!

Now while vector notation is neat, it's actually really unhelpful when we wish to do calculus. Each Jacobian in the above expression is a matrix and I always forget how to order the rows and columns properly. Furthermore, the following is going to involve a lot of vector derivatives, matrix derivatives, and higher order tensor derivatives, which can all be very unwieldy, so to ease notation we shall adopt index notation instead. As we shall see, switching up our notation frequently is going to help our understanding and aid our ability to generalize.

So using $z^i$ to denote the $i$th component of a vector $\textbf{z}$, we could write
\begin{align}
    \frac{\partial z^i}{\partial x^j} = \sum_{\alpha} \frac{\partial z^i}{\partial y^\alpha} \frac{\partial y^\alpha}{\partial x^j}.
\end{align}
As a second notational step, we are going to denote differentiation of a function $h$ with respect to the $\alpha$th dimension of its input as $h_\alpha$. Notice we do not need to make  reference to $y$ in this notation, since it is understood at we differentiate with respect to the input of $f$, however we might wish to label it. So
\begin{align}
    \frac{\partial f^i}{\partial y^\alpha} = f^i_\alpha
\end{align}
The chain rule is then just
\begin{align}
    \frac{\partial f^i}{\partial x^j} = \sum_{\alpha} f^i_\alpha g^\alpha_j.
\end{align}
Notice how there is one $\alpha$ on the bottom and one $\alpha$ on the top. For this reason, as one final notational convenience, we will switch to Einstein notation, where we implicitly sum over repeated indices in upper--lower pairs, so the chain rule is
\begin{align}
    \frac{\partial f^i}{\partial x^j} =  f^i_\alpha g^\alpha_j.
\end{align}
I have always found this notation both very elegant and parsimonious. Back in my PhD, before automatic differentiation was commonplace in machine learning, I would often use this notation to work out gradients, because it is both uncluttered and unconfusing.

You may have noticed that I am using Greek letters for the dummy variables we sum over. This is just a choice mainly for me to remember what we are summing over. With this highly compressed notation, let's write the $2$nd derivative of $f^i$ with respect to $x$. It's
\begin{align}
    \frac{\partial^2 f^i}{\partial x^j \partial x^k} = f^i_{\alpha \beta} g^\alpha_j g^\beta_k + f^i_{\alpha} g^\alpha_{jk}.
\end{align}
The 3th derivative is
\begin{align}
    \frac{\partial^3 f^i}{\partial x^j \partial x^k \partial x^\ell} &= f^i_{\alpha \beta \gamma} g^\alpha_j g^\beta_k g^\gamma_\ell + f^i_{\alpha \beta} g^\alpha_{j\ell} g^\beta_k + + f^i_{\alpha \beta} g^\alpha_{j} g^\beta_{k\ell} + f^i_{\alpha \beta} g^\alpha_{jk} g^\beta_\ell + f^i_{\alpha} g^\alpha_{jk\ell} \newline
    &= f^i_{\alpha \beta \gamma} g^\alpha_j g^\beta_k g^\gamma_\ell + 3 \cdot  f^i_{\alpha \beta} g^\alpha_j g^\beta_{k\ell} + f^i_{\alpha} g^\alpha_{jk\ell}
\end{align}
These expressions get very unwieldy for higher order derivatives. Let's try one fourth!
\begin{align}
    \frac{\partial^4 f^i}{\partial x^j \partial x^k \partial x^\ell \partial x^m} &= f^i_{\alpha \beta \gamma \delta} g^\alpha_j g^\beta_k g^\gamma_\ell g^\delta_m + 6 \cdot f^i_{\alpha \beta \gamma} g^\alpha_j g^\beta_k g^\gamma_{\ell m} + 3 \cdot  f^i_{\alpha \beta} g^\alpha_{j\ell} g^\beta_{km}
    + 4 \cdot  f^i_{\alpha \beta} g^\alpha_{j} g^\beta_{k \ell m} + f^i_{\alpha} g^\alpha_{jk\ell m}.
\end{align}
OK, what is going on? This is tedious and confusing and it is not obvious if there is any structure to this. In fact there is a very simple structure and we can derive all the above with some simple rules involving *special labeled rooted trees*. To make the connection, we make two observations. Each derivative is a sum of factors of the form $f^i_{\alpha\beta...}g^\alpha_{ij...}g^\beta_{k\ell...} \cdots$ where there is a:

1. single term in $f^i_{\alpha\beta...}$ with multiple subscripts,
2. multiple terms in $g^\alpha_{ij...}$ where $g$ has a single superscript and potentially many subscripts.

We are going to replace each term in $f$ or $g$ with parts of a special rooted tree.

# Special labeled rooted trees
We begin by drawing the simplest tree $f^i$ as
<p align="center">
  <img src="/media/2023/aod_1.svg">
</p>
This is just a root node of a tree---hence special labeled *rooted* tree. Every time we differentiate $f^i$ we will draw a branch emanating from the root node. In other words, for every subscript of $f^i$ we draw a branch. The first derivative $f^i_{\alpha} g^\alpha_j$ we thus draw as

<p align="center">
  <img src="/media/2023/aod_2.svg">
</p>

This is simple enough. Note, we shall also label the nodes with the subscript of the attached branch---in this case $j$---so that we can keep track of what branch corresponds to what algebraïc terms. Hence special *labeled* rooted tree. We didn't write $i$ by the root node, since it is not a *sub*script. In fact, since $i$ only ever appears in the superscript of $f$, we could drop it entirely, leaving $f$ as a vector-in scalar-out mapping, which we choose to do from now on.

Now what about the factor $f_{\alpha\beta} g^\alpha_j g^\beta_k$? It has two branches emanating from the root as

<p align="center">
  <img src="/media/2023/aod_3.svg">
</p>

What if $g$ has multiple subscripts? Well, we then extend the branch by as many subscripts in $g$ so $f_{\alpha} g^\alpha_{jk}$ and $f_{\alpha\beta} g^\alpha_{jk}g^\beta_\ell$ look like

<p align="center">
  <img src="/media/2023/aod_4.svg">
</p>

This notation is a little weird at first, but as expressions get longer and more cumbersome, the tree representations become easier to handle. Now we have everything we need to differentiate the tree representation of our function $f(g(\textbf{x}))$. The $1$st derivative of $f$ is $f_\alpha g^\alpha_j$, which is a single branched tree

<p align="center">
  <img src="/media/2023/aod_5.svg">
</p>

I have drawn the new branch in red to emphasize it. Differentiating again yields $f_{\alpha \beta} g^\alpha_j g^\beta_k + f_{\alpha} g^\alpha_{jk}$, so

<p align="center">
  <img src="/media/2023/aod_6.svg">
</p>

What just happened? When differentiating $f_\alpha g^\alpha_j$, which in the literature is called an *elementary differential*, we applied the product rule and made two copies of $f_\alpha g^\alpha_j$. To the first copy we differentiated the $f_{\alpha}$ term, adding a new subscript $\beta$ and an extra $g^\beta_k$ branch to the root. To the second copy we differentiated the $g^\alpha_j$ term, raising it to a $2$nd order deriviative, and thus extending the already existing $g^\alpha_j$ branch to a length $2$ $g^\alpha_{jk}$.

We can easily see how this technique generalizes to higher order factors. We apply the product rule and make as many copies of our special labeled rooted tree as there are terms in the factor. To the first copy we add a branch corresponding to differentiating $f$ and to the remaining copies we extend each of the existing branches, one by one. Let's apply this technique to differentiate again, either adding a new branch to root or extending existing branches. This yields

<p align="center">
  <img src="/media/2023/aod_7.svg">
</p>

Now, noticing that the middle three trees are topologically the same, with permuted labels, we can rewrite this, but we need to strip the labels. This results in

<p align="center">
  <img src="/media/2023/aod_8.svg">
</p>

which corresponds to the expression $f_{\alpha \beta \gamma} g^\alpha_j g^\beta_k g^\gamma_\ell + 3 \cdot  f_{\alpha \beta} g^\alpha_j g^\beta_{k\ell} + f_{\alpha} g^\alpha_{jk\ell}$ that we derived earlier! These new label-less trees are referred to as simply as *special rooted trees*. In maths-speak, a special rooted tree is an representative of the equivalence class of special labeled rooted trees.

# Aside: Where does that 3 come from?
That 3 we see popping up in front is the *cardinality* of the equivalence class--the total number of valid labelings of the tree. Without getting too distracted, for a labeling to be valid labels need to increase from the root, so

<p align="center">
  <img src="/media/2023/aod_9.svg">
</p>

is an invalid labeling, assuming we have chosen alphabetical ordering of labels. On the surface, it's not very obvious why the coefficients that precede the elementary differentials in higher derivative expressions would naturally be the number of valid labelings. But staring at the diagram of how we differentiate special labeled rooted trees, we see that each row essentially generated all possible special rooted labeled trees. So all possible labelings of each special rooted labeled tree are enumerated. And hence these coefficients have a very beautiful origin.

For those with a background in combinatorix, you will probably be quick to realize that there is a bijection between special rooted labeled trees and integer partitions of sets. We can associate each of the following  4-node trees with partitions with integer partitions of the set $\{j, k, \ell\}$

<p align="center">
  <img src="/media/2023/aod_10.svg">
</p>

Each branch in the diagram is a grouping of letters into a subset. While each branch has to be ordered alphabetically from its root, there is only one such valid ordering, so the subset can just be left unordered. We could go deeper into partitions of sets, but Wikipedia is your friend here.

# Back to differentiation
For me I would say the tree representation is much easier to parse than the algebraïc representation, which, mind you, is still shorthand for
\begin{align}
    \frac{\partial^3 f}{\partial y^\alpha \partial y^\beta \partial y^\gamma}\frac{\partial g^\alpha}{\partial y^j}\frac{\partial g^\beta}{\partial y^k}\frac{\partial g^\gamma}{\partial y^\ell} + 3\frac{\partial^2 f}{\partial y^\alpha \partial y^\beta}\frac{\partial g^\alpha}{\partial y^j}\frac{\partial^2 g^\beta}{\partial y^k \partial y^\ell}+ \frac{\partial f}{\partial y^\alpha}\frac{\partial^3 g^\alpha}{\partial y^j \partial y^k \partial y^\ell}.
\end{align}
What would be the expression for the $5$th order derivative?

So we can study higher order derivatives of compositions of functions via special rooted trees! This process of adding and extending branches can be applied recursively very easily and a list of the first few special rooted trees looks like

<p align="center">
  <img src="/media/2023/aod_11.svg">
</p>

The theory of rooted trees goes very deep. We have only considered the *special* variety, for which branching can only occur at the root node. People have gone far into defining entire algebras over rooted trees, defining operations such as multiplication and addition. This comes in handy when studying order conditions of Runge-Kutta solvers and renormalization in quantum field theory. I personally think this area is extremely beautiful and am even more happy that I have a quick trick to derive expressions for higher order derivatives of composed functions.
