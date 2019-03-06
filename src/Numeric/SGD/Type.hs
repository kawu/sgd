-- | Provides the basic `SGD` pipe type.


module Numeric.SGD.Type
  ( SGD
  ) where


import Pipes as P


-- | SGD is a pipe which, given the initial parameter values, consumes training
-- elements of type @e@ and outputs the subsequently calculated parameter sets
-- of type @p@.
type SGD m e p = p -> P.Pipe e p m ()
