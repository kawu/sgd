{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Implementation of SGD with momentum


module Numeric.SGD.Momentum
  ( Config(..)
  , Args(..)
  , momentum
  , momentumM
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (when)

import qualified Pipes as P

import           Numeric.SGD.ParamSet
import           Numeric.SGD.Args


-- | Momentum configuration
data Config = Config
  { gain0 :: Double
  -- ^ Initial gain parameter
  , tau :: Double
  -- ^ After how many gradient calculations the gain parameter is halved
  , gamma :: Double
  -- ^ Exponentional decay parameter
  } deriving (Show, Eq, Ord, Generic)


-- | General, monadic stochastic gradient descent with momentum.
momentumM
  :: (Monad m, ParamSet p)
  => Config
    -- ^ Momentum configuration
  -> Args m p
    -- ^ General SGD arguments
  -> p 
    -- ^ Initial parameters
  -> m p
momentumM Config{..} Args{..} net0 =

  go 0 (zero net0) net0

  where

    -- Gain in the k-th iteration
    gain k
      = (gain0 * tau)
      / (tau + fromIntegral k)

    go k moment net
      | k > iterNum = return net
      | otherwise = do
          when (k `mod` reportPeriod == 0) $ do
            report net
          grad <- scale (gain k) <$> gradient net
          let moment' = scale gamma moment `add` grad
              newNet = net `sub` moment'
          go (k+1) moment' newNet


-- | Stochastic gradient descent with momentum.
momentum
  :: (Monad m, ParamSet p)
  => Config
    -- ^ Momentum configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element.
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
