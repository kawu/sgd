{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Stochastic gradient descent with momentum, following:
--
--     * http://ruder.io/optimizing-gradient-descent/index.html#momentum


module Numeric.SGD.Momentum
  ( Config(..)
  , def
  , momentum
  ) where


import           GHC.Generics (Generic)

import qualified Pipes as P

import           Numeric.SGD.ParamSet


-- | Momentum configuration
data Config = Config
  { gain0 :: Double
  -- ^ Initial gain parameter, used to scale the gradient
  , tau :: Double
  -- ^ After how many gradient calculations the gain parameter is halved
  , gamma :: Double
  -- ^ Momentum term
  } deriving (Show, Eq, Ord, Generic)


-- | Default momentum configuration
def :: Config
def = Config
  { gain0 = 0.01
  , gamma = 0.9
  , tau = 1000
  }


-- | Stochastic gradient descent with momentum.
momentum
  :: (Monad m, ParamSet p)
  => Config
    -- ^ Momentum configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element
  -> p 
    -- ^ Initial parameters
  -> P.Pipe e p m ()
momentum Config{..} gradient net0 =

  go (0 :: Integer) (zero net0) net0

  where

    -- Gain in the k-th iteration
    gain k
      = (gain0 * tau)
      / (tau + fromIntegral k)

    go k moment net = do
      x <- P.await
      let grad = scale (gain k) (gradient x net)
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
