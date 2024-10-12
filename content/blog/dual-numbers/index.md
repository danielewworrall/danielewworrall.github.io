---
title: "Dual numbers"
date: 2021-08-09                    
tags:
  - optimisation
  - differentiation
  - automatic differentiation
  - algebra
extract: There is a generalisation of the complex numbers where {{< math >}}$i^2=0${{< math >}}
---
# Dual numbers I

**TL;DR**: There is a generalisation of the complex numbers where {{< math >}}$i^2=0${{< math >}} instead of {{< math >}}$i^2=-1${{< math >}}. Functions extended to this _dual number_ system have the curious property that we can read off their derivatives (at a point {{< math >}}$x$) if we evaluate them at the dual number {{< math >}}$x + i${{< math >}}. This has implications for automatic differentiation frameworks.

I found writing this next post is a real treat. It's about *dual numbers*. Dual numbers are a bit strange, to say the least, and at first they seem like an abstract mathematical fancy, but as you will see they serve quite a useful purpose in the realm of automatic differentiation. We're going to start by reviewing what we know about the complex numbers. It will turn out that by tweaking them a wee bit we end up with the dual numbers, which, as mentioned, have some strikingly elegant properties when it comes to evaluating derivatives on computation graphs.

### Complex numbers
The complex numbers {{< math >}}$z \in \mathbb{C}${{< math >}} are typically expressed in split real-imaginary form {{< math >}}$z = a + ib${{< math >}}, where {{< math >}}$a, b \in \mathbb{R}${{< math >}} are real numbers and {{< math >}}$i${{< math >}} is the *imaginary unit*. In high school and as a young engineering undergraduate student these were the bane of my life. {{< math >}}$i${{< math >}} has this weird property that {{< math >}}$i^2 = -1${{< math >}}. Apart from that though, the complex numbers seem to act just like the reals under algebraic manipulation, so if {{< math >}}$z = a + ib${{< math >}} and {{< math >}}$y = c + i d${{< math >}}, then

{{< math >}}
$$
    z y = (a + ib) (c + id) = ac + iad + ibc + \color{red}{i^2}bd = (ac - bd) + i(ad + bc).
$$
{{< /math >}}

Now why did I have to learn to perform these mundane manipulations? Well the complex numbers have these beautiful geometric properties that connect them with the trigonometric functions. Since (periodic and analytic) functions can be expanded in a trigonometric basis, it turned out that we could study just about any function of interest in the complex domain and usually it was simpler to do so.

### Hypercomplex numbers: complex, double, and dual
But do we necessarily have to demand that {{< math >}}$i^2 = -1$? Well no. In fact, allowing {{< math >}}$i^2${{< math >}} to equal other values opens up a garden of delights. If we set {{< math >}}$i^2 = 1${{< math >}} we have the *double numbers*, also known as the *split complex-numbers*, and if we set {{< math >}}$i^2 = 0${{< math >}} (making sure that {{< math >}}$i \neq 0$), then we have the *dual numbers*. It turns out that all 3 number systems for {{< math >}}$i^2=-1${{< math >}}, {{< math >}}$i^2=1${{< math >}}, and {{< math >}}$i^2=0${{< math >}} are cases of [hypercomplex numbers](https://en.wikipedia.org/wiki/Hypercomplex_numbers). The above multiplication for dual numbers is

{{< math >}}
$$
    z y = (a + ib) (c + id) = ac + iad + ibc + \color{red}{i^2}bd = ac + i(ad + bc).
$$
{{< /math >}}

The term {{< math >}}$i^2bd${{< math >}} falls away since {{< math >}}$i^2=0${{< math >}}, as we defined. A mathematical object, where {{< math >}}$\underbrace{ii \cdots i}_{k \text{ times}} = 0${{< math >}} is called *nilpotent* with degree {{< math >}}$k${{< math >}}.

### Taylor expansions and exact linearisation
Let's do some computations and see why dual numbers are useful. We are going to take a function {{< math >}}$f${{< math >}} defined on the real domain and stick dual numbers {{< math >}}$z = x + iy${{< math >}} into it. This may seem cowboyish, and it is, but it will lead us somewhere very satisfying. Now how do we evaluate a function at a point {{< math >}}$x + iy$? Well if the function is [analytic](https://en.wikipedia.org/wiki/Analytic_function) (fancy word for smooth), then we can just use a Taylor expansion, so

{{< math >}}
$$
\begin{align}
    f(x + iy) &= f(x) + iy f'(x) + \underbrace{\frac{(iy)^2}{2!}}_{=0} f''(x) + \underbrace{\frac{(iy)^3}{3!}}_{=0} f'''(a) + ... \newline
    &= f(x) + iy f'(x).
\end{align}
$$
{{< /math >}}

We dropped the second-order and higher-order terms because they all contained terms with {{< math >}}$i^2${{< math >}}, which we have defined as zero. This is marvellous! By evaluating {{< math >}}$f${{< math >}} at the point {{< math >}}$x + iy${{< math >}}, we can return its exact linearisation. No need for Big-$\mathcal{O}${{< math >}} Notation and hand-waving about {{< math >}}$iy${{< math >}} being 'small enough'. Furthermore, if we set {{< math >}}$y=1${{< math >}}, then we can read off the derivative of {{< math >}}$f${{< math >}} as the dual component (analogous to imaginary component) of {{< math >}}$f(x + i)${{< math >}}.

Regarding terminology, for a dual number {{< math >}}$x+iy${{< math >}} it is common to call {{< math >}}$x${{< math >}} the *primal* since it represents the primary component of the computation; {{< math >}}$y${{< math >}} is the *tangent*, giving a nod to the fact that it. represents a derivative, which lives in a tangent space; and {{< math >}}$i${{< math >}} is the tag, which is an odd name, but it will make sense in a following blog when I discuss higher-order derivatives.

### Computation graphs and the chain rule
In modern machine learning, we like to build composable functions and optimise all the parameters using automatic differentiation. Automatic differentiation is just a souped-up version of the chain rule. Let's see how dual numbers pair with the chain rule. First of all a recap of the chain rule:

{{< math >}}
$$
    \frac{\mathrm{d}}{\mathrm{d} x} f(g(x)) = \color{red}{f'(g(x)) g'(x)}.
$$
{{< /math >}}

Now with dual numbers

{{< math >}}
$$
    f(g(x + i)) = f(g(x) + ig'(x)) = f(g(x)) + i\color{red}{f'(g(x)) g'(x)}.
$$
{{< /math >}}

So we see indeed that the tangent component of {{< math >}}$f(g(x + i))${{< math >}} is indeed the correct derivative, had we used the chain rule! How would we code this? How do we even represent dual numbers?

### Automagic dual number-based differentation
What do we need to implement dual number-based automatic differentation? First we need a dictionary of composible atomic functions {{< math >}}$f_1, f_2, f_3, ...${{< math >}}, typically called *primitives*, from which we can build a computation graph. All deep learning libraries contain them. For instance, think of PyTorch's `torch.nn.functional.relu()`. Next we are going to require all of their derivative functions. Just like in standard backprop, we always have these at hand. Typically we may be used to defining a function with separate `forward()` and `backward()` methods for the evaluation and derivative separately. For instance, consider the tangent function {{< math >}}$f(x) = \tan(x)${{< math >}}, which has derivative {{< math >}}$f'(x) = 1 + \tan^2(x)$:

```python
class Tan:
    def forward(self, x):
        return np.tan(x)

    def backward(self, x):
        return 1 + (np.tan(x) * np.tan(x))
```

Differentiation by dual numbers works differently by overloading the arguments to a function {{< math >}}$f${{< math >}}, using dual numbers instead of real ones. Likewise, we can take a function, and overload its input. The alternative is to define a separate function `dtan()` with the desired properties. This method is called *source code transformation*. We can represent dual number {{< math >}}$z = x + iy${{< math >}} as a tuple {{< math >}}$(x, y)${{< math >}}. Then

```python
def dtan(z):
    x, y = z
    return Tan().forward(x), y * Tan().backward(x)
```

That's it. At the start of your computational graph, just specify {{< math >}}$z=(x,1)${{< math >}} and away you go! A more modular way to implement this would be to specify a dual method that can operate on any function, not just `Tan()`. This would have form

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

Now the real test is to check whether composition works. Let's take the derivative of {{< math >}}$f(x) = \tan(\tan(x))${{< math >}}, where

{{< math >}}
$$
    \frac{\mathrm{d} f}{\mathrm{d} x} = (1 + \tan^2(\tan(x)))\cdot(1 + \tan^2(x)).
$$
{{< /math >}}

Evaluating this at {{< math >}}$x=2${{< math >}}, in code this is

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
