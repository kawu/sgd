# Haskell stochastic gradient descent library

Stochastic gradient descent (SGD) is a method for optimizing a global objective
function defined as a sum of smaller, differentiable functions.  In each
iteration of SGD the gradient is calculated based on a subset of the training
dataset.  In Haskell, this process can be simply represented as a [fold over a
of subsequent dataset
subsets](https://blog.jle.im/entry/purely-functional-typed-models-1.html)
(singleton elements in the extreme).

However, it can be beneficial to select the subsequent subsets randomly (e.g.,
shuffle the entire dataset before each pass).  Moreover, the dataset can be
large enough to make it impractical to store it all in memory.  Hence, the
`sgd` library adopts a [pipe](http://hackage.haskell.org/package/pipes)-based
interface in which SGD takes the form of a process consuming dataset subsets
(the so-called mini-batches) and producing a stream of output parameter values.

The `sgd` library implements several SGD variants (SGD with momentum, AdaDelta,
Adam) and handles heterogeneous parameter representations (vectors, maps, custom
records, etc.).  It can be used in combination with automatic differentiation
libraries ([ad](http://hackage.haskell.org/package/ad),
[backprop](http://hackage.haskell.org/package/backprop)), which can be used to
automatically calculate the gradient of the objective function.

Look at [the hackage repository](http://hackage.haskell.org/package/sgd) for a
library documentation.
