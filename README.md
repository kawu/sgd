# Stochastic gradient descent library

* k -- step of the gradient descent procedure; at each step, `batchSize` dataset
  elements are used to estimate the gradient to follow
* v<sub>k</sub> -- vector of model parameters (k-th step)
* u<sub>k</sub> -- gradient vector (k-th step)
* s<sub>k</sub> -- step size, calculated on the basis of `gain0` (initial step
  size) and `tau` arguments (it decreases to `gain0` / 2 after `tau` passes
  over the full dataset)
* `regVar` -- regularization variance

# No regularization

The parameters for the step k+1 are computed as follows:
* v<sub>k+1</sub> = v<sub>k</sub> + s<sub>k</sub> * u<sub>k</sub>

Note that the above equation relies on the fact that the gradient computed over
the entire dataset is equal to the sum of the gradients computed separately for
the individual sentences in this dataset.

# With regularization

Regularization parameter after full dataset pass:
* r<sub>k</sub> = 1 - (s<sub>k</sub> / `regVar`),
where s<sub>k</sub> is used to gradually decrease the effect of regularization,
just as the effect of the gradient u<sub>k</sub> is controlled (see above).

Each full dataset pass, regularization could be applied: 
* v<sub>k</sub> *= r<sub>k</sub>

However, we rather apply it every step, which yields:
* v<sub>k</sub> *= r<sub>k</sub> ^ `coef`,
where `coef` = `batchSize` / size of the full dataset.
