---
title: "Dual numbers"
date: 2021-08-09                    
permalink: /blog/2021/08/dual-numbers/
tags:
  - optimisation
  - differentiation
  - automatic differentiation
  - algebra
excerpt: "I found writing this next post is a real treat. It's about *dual numbers*. Dual numbers are a bit strange, to say the least, and at first they seem like an abstract mathematical fancy, but as you will see they have quite a useful purpose in the realm of automatic differentiation."
---
# Dual numbers I

**TL;DR**: There is a generalisation of the complex numbers where $i^2=0$ instead of $i^2=-1$. Functions extended to this _dual number_ system have the curious property that we can read off their derivatives (at a point $x$) if we evaluate them at the dual number $x + i$. This has implications for automatic differentiation frameworks.

I found writing this next post is a real treat. It's about *dual numbers*. Dual numbers are a bit strange, to say the least, and at first they seem like an abstract mathematical fancy, but as you will see they serve quite a useful purpose in the realm of automatic differentiation. We're going to start by reviewing what we know about the complex numbers. It will turn out that by tweaking them a wee bit we end up with the dual numbers, which, as mentioned, have some strikingly elegant properties when it comes to evaluating derivatives on computation graphs.

### Complex numbers
The complex numbers $z \in \mathbb{C}$ are typically expressed in split real-imaginary form $z = a + ib$, where $a, b \in \mathbb{R}$ are real numbers and $i$ is the *imaginary unit*. In high school and as a young engineering undergraduate student these were the bane of my life. $i$ has this weird property that $i^2 = -1$. Apart from that though, the complex numbers seem to act just like the reals under algebraic manipulation, so if $z = a + ib$ and $y = c + i d$, then

$$
    z y = (a + ib) (c + id) = ac + iad + ibc + \color{red}{i^2}bd = (ac - bd) + i(ad + bc).
$$

Now why did I have to learn to perform these mundane manipulations? Well the complex numbers have these beautiful geometric properties that connect them with the trigonometric functions. Since (periodic and analytic) functions can be expanded in a trigonometric basis, it turned out that we could study just about any function of interest in the complex domain and usually it was simpler to do so.

### Hypercomplex numbers: complex, double, and dual
But do we necessarily have to demand that $i^2 = -1$? Well no. In fact, allowing $i^2$ to equal other values opens up a garden of delights. If we set $i^2 = 1$ we have the *double numbers*, also known as the *split complex-numbers*, and if we set $i^2 = 0$ (making sure that $i \neq 0$), then we have the *dual numbers*. It turns out that all 3 number systems for $i^2=-1$, $i^2=1$, and $i^2=0$ are cases of [hypercomplex numbers](https://en.wikipedia.org/wiki/Hypercomplex_numbers). The above multiplication for dual numbers is

$$
    z y = (a + ib) (c + id) = ac + iad + ibc + \color{red}{i^2}bd = ac + i(ad + bc).
$$

The term $i^2bd$ falls away since $i^2=0$, as we defined. A mathematical object, where $\underbrace{ii \cdots i}_{k \text{ times}} = 0$ is called *nilpotent* with degree $k$.

### Taylor expansions and exact linearisation
Let's do some computations and see why dual numbers are useful. We are going to take a function $f$ defined on the real domain and stick dual numbers $z = x + iy$ into it. This may seem cowboyish, and it is, but it will lead us somewhere very satisfying. Now how do we evaluate a function at a point $x + iy$? Well if the function is [analytic](https://en.wikipedia.org/wiki/Analytic_function) (fancy word for smooth), then we can just use a Taylor expansion, so

$$ \begin{aligned}
    f(x + iy) &= f(x) + iy f'(x) + \underbrace{\frac{(iy)^2}{2!}}_{=0} f''(x) + \underbrace{\frac{(iy)^3}{3!}}_{=0} f'''(a) + ... \newline
    &= f(x) + iy f'(x).
  \end{aligned}
$$

We dropped the second-order and higher-order terms because they all contained terms with $i^2$, which we have defined as zero. This is marvellous! By evaluating $f$ at the point $x + iy$, we can return its exact linearisation. No need for Big-$\mathcal{O}$ Notation and hand-waving about $iy$ being 'small enough'. Furthermore, if we set $y=1$, then we can read off the derivative of $f$ as the dual component (analogous to imaginary component) of $f(x + i)$.

Regarding terminology, for a dual number $x+iy$ it is common to call $x$ the *primal* since it represents the primary component of the computation; $y$ is the *tangent*, giving a nod to the fact that it. represents a derivative, which lives in a tangent space; and $i$ is the tag, which is an odd name, but it will make sense in a following blog when I discuss higher-order derivatives.

### Computation graphs and the chain rule
In modern machine learning, we like to build composable functions and optimise all the parameters using automatic differentiation. Automatic differentiation is just a souped-up version of the chain rule. Let's see how dual numbers pair with the chain rule. First of all a recap of the chain rule:

$$
    \frac{\mathrm{d}}{\mathrm{d} x} f(g(x)) = \color{red}{f'(g(x)) g'(x)}.
$$

Now with dual numbers

$$
    f(g(x + i)) = f(g(x) + ig'(x)) = f(g(x)) + i\color{red}{f'(g(x)) g'(x)}.
$$

So we see indeed that the tangent component of $f(g(x + i))$ is indeed the correct derivative, had we used the chain rule! How would we code this? How do we even represent dual numbers?

### Automagic dual number-based differentation
What do we need to implement dual number-based automatic differentation? First we need a dictionary of composible atomic functions $f_1, f_2, f_3, ...$, typically called *primitives*, from which we can build a computation graph. All deep learning libraries contain them. For instance, think of PyTorch's `torch.nn.functional.relu()`. Next we are going to require all of their derivative functions. Just like in standard backprop, we always have these at hand. Typically we may be used to defining a function with separate `forward()` and `backward()` methods for the evaluation and derivative separately. For instance, consider the tangent function $f(x) = \tan(x)$, which has derivative $f'(x) = 1 + \tan^2(x)$:

```python
class Tan:
    def forward(self, x):
        return np.tan(x)

    def backward(self, x):
        return 1 + (np.tan(x) * np.tan(x))
```

Differentiation by dual numbers works differently by overloading the arguments to a function $f$, using dual numbers instead of real ones. Likewise, we can take a function, and overload its input. The alternative is to define a separate function `dtan()` with the desired properties. This method is called *source code transformation*. We can represent dual number $z = x + iy$ as a tuple $(x, y)$. Then

```python
def dtan(z):
    x, y = z
    return Tan().forward(x), y * Tan().backward(x)
```

That's it. At the start of your computational graph, just specify $z=(x,1)$ and away you go! A more modular way to implement this would be to specify a dual method that can operate on any function, not just `Tan()`. This would have form

```python
def dual(primitive):
    def df(z):
        x, y = z
        return primitive.forward(x), y * primitive.backward(x)
    return df
```

Then `dtan` is equivalent to `dual(Tan())`. For instance

```python
dtan = dual(Tan())

print(dtan((2, 1)))
>>> (-2.185039863261519, 5.774399204041917)
print((np.tan(2), 1+np.tan(2)**2))
>>> (-2.185039863261519, 5.774399204041917)
```

Now the real test is to check whether composition works. Let's take the derivative of $f(x) = \tan(\tan(x))$, where

$$
    \frac{\mathrm{d} f}{\mathrm{d} x} = (1 + \tan^2(\tan(x)))\cdot(1 + \tan^2(x)).
$$

Evaluating this at $x=2$, in code this is

```python
print(dtan(dtan((2, 1))))
>>> (1.417928575505387, 17.383952637114582)

print((np.tan(np.tan(2)), (1+np.tan(np.tan(2))**2)*(1+np.tan(2)**2)))
>>> (1.417928575505387, 17.383952637114582)
```

It really is amazing how embarrasingly simple this technique is. Notice how the order of gradient call executions is the same as the order of the forward pass. As such, this is an incarnation of so-called *forward-mode differention*, to be contrasted with *reverse-mode differentation*, which machine learners tend to call *backprop*.

Why don't deep learners use this method then, if it's so simple? The answer is that for high dimensions, the forward accumulated gradient is a Jacobian, which can be a prohibitively large matrix; whereas, the backpropagated gradient is a vector.

### Next steps
The method above is sweet and simple, but it has some failings when we wish to do slightly more complicated things, such as evaluate higher-order derivatives. In the next post, I'll show a slightly more sophisticated way to implement dual number-based differentiation, which side-steps these issues.



### References
- https://www.mdpi.com/2075-1680/8/3/77/html
- http://blog.jliszka.org/2013/10/24/exact-numeric-nth-derivatives.html
- https://en.wikipedia.org/wiki/Dual_number
- https://encyclopediaofmath.org/wiki/Double_and_dual_numbers
- https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
