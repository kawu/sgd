# Stochastic gradient descent library

* k -- step of the gradient descent procedure; at each step, `batchSize` dataset
  elements are used to estimate the gradient to follow
* v<sub>k</sub> -- vector of model parameters (k-th step)
* u<sub>k</sub> -- gradient vector (k-th step)
* s<sub>k</sub> -- step size (calculated on the basis of the initial step size
  `gain0` and `tau` arguments)
* `regVar` -- regularization variance

# No regularization

The parameters for the step k+1 are computed as follows:
* v<sub>k+1</sub> = v<sub>k</sub> + s<sub>k</sub> * u<sub>k</sub>

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
