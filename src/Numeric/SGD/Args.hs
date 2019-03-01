-- | General SGD arguments


module Numeric.SGD.Args
  ( Args(..)
  ) where


-- | SGD arguments
data Args m p = Args
  { iterNum :: Int
    -- ^ Number of iterations
  , gradient :: p -> m p
    -- ^ Gradient on (some part of) the training data.  The function is monadic
    -- because of its stochasticit nature.
  , reportPeriod :: Int
    -- ^ Run `report` every `reportPeriod` iterations
  , report :: p -> m ()
    -- ^ Reporting function. It can be used to store the intermediary parameter
    -- values or print the value of the objective function on the training
    -- dataset. 
  }
