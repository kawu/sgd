{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Provides the `momentum` function which implements stochastic gradient
-- descent with momentum, following:
--
--     * http://ruder.io/optimizing-gradient-descent/index.html#momentum


module Numeric.SGD.Momentum
  ( Config(..)
  , scaleTau
  , momentum
  ) where


import           GHC.Generics (Generic)

import           Data.Default

import qualified Pipes as P

import           Numeric.SGD.Type
import           Numeric.SGD.ParamSet


-- | Momentum configuration
data Config = Config
  { alpha0 :: Double
    -- ^ Initial step size, used to scale the gradient
  , tau :: Double
    -- ^ The step size after k * `tau` iterations = `alpha0` / (k + 1)
  , gamma :: Double
    -- ^ Momentum term
  } deriving (Show, Eq, Ord, Generic)

instance Default Config where
  def = Config
    { alpha0 = 0.01
    , gamma = 0.9
    , tau = 1000
    }


-- | Scale the `tau` parameter.  Useful e.g. to account for the size of the
-- training dataset.
scaleTau :: Double -> Config -> Config
scaleTau coef cfg = cfg {tau = coef * tau cfg}


-- | Stochastic gradient descent with momentum. See "Numeric.SGD.Momentum" for
-- more information.
momentum
  :: (Monad m, ParamSet p)
  => Config
    -- ^ Momentum configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element
  -> SGD m e p
momentum Config{..} gradient net0 =

  go (0 :: Integer) (zero net0) net0

  where

    -- Gain in the k-th iteration
    alpha k
      = (alpha0 * tau)
      / (tau + fromIntegral k)

    go k moment net = do
      x <- P.await
      let grad = scale (alpha k) (gradient x net)
          moment' = scale gamma moment `add` grad
          newNet = net `sub` moment'
      P.yield newNet
      go (k+1) moment' newNet


-------------------------------
-- Utils
-------------------------------


-- | Scaling
scale :: ParamSet p => Double -> p -> p
scale x = pmap (*x)
{-# INLINE scale #-}
