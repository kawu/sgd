{-# LANGUAGE RecordWildCards #-}


module Numeric.SGD.ParamSet
  ( ParamSet(..)
  ) where


import           Prelude hiding (div)


-- | Class of types that can be treated as parameter sets.  It provides basic
-- element-wise opertations (addition, multiplication, mapping) on parameter
-- sets which are required to perform stochastic gradient descent.  Many of the
-- operations (`add`, `mul`, `sub`, `div`, etc.) have the same interpretation
-- and follow the same laws (e.g. associativity) as the corresponding
-- operations in `Num` and `Fractional`.  Objects of this class can be also
-- seen as a containers of parameters, hence `pmap` which can be seen as a
-- monomorphic version of `fmap`.
--
-- Minimal complete definition: `zero`, `pmap`, (`add` or `sub`), and (`mul` or
-- `div`).
--
-- If you leave the body of an instance declaration blank, GHC Generics will be
-- used to derive instances if the type has a single constructor and each field
-- is an instance of `ParamSet`.
class ParamSet p where
  -- | Element-wise zero (the additive identity)
  zero :: p
  -- | Element-wise mapping
  pmap :: (Double -> Double) -> p -> p

  -- | Element-wise negation
  neg :: p -> p
  neg = pmap (\x -> -x)
  -- | Element-wise addition
  add :: p -> p -> p
  add x y = x `sub` neg y
  -- | Elementi-wise substruction
  sub :: p -> p -> p
  sub x y = x `add` neg y

  -- | Element-wise multiplication
  mul :: p -> p -> p
  mul x y = x `div` pmap (1.0/) y
  -- | Element-wise division
  div :: p -> p -> p
  div x y = x `mul` pmap (1.0/) y


-- -- | Root square
-- squareRoot :: p -> p
-- squareRoot = pmap sqrt
-- 
-- -- | Square
-- square :: p -> p
-- square x = x `mul` x
--
-- -- | Scaling
-- scale :: Double -> p -> p
-- scale x = pmap (*x)
