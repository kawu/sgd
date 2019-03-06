{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Provides the `adaDelta` function which implements the AdaDelta algorithm
-- as described in the following paper:
--
--     * https://arxiv.org/pdf/1212.5701.pdf


module Numeric.SGD.AdaDelta
  ( Config(..)
  , adaDelta
  ) where


import           GHC.Generics (Generic)

import           Prelude hiding (div)
-- import           Control.Monad (when)

import           Data.Default

import qualified Pipes as P

import           Numeric.SGD.Type
import           Numeric.SGD.ParamSet
-- import           Numeric.SGD.Args


-- | AdaDelta configuration
data Config = Config
  { decay :: Double
    -- ^ Exponential decay parameter
  , eps   :: Double
    -- ^ Epsilon value
  } deriving (Show, Eq, Ord, Generic)

instance Default Config where
  def = Config
    { decay = 0.9
    , eps = 1.0e-6
    }


-- | Perform gradient descent using the AdaDelta algorithm.  
-- See "Numeric.SGD.AdaDelta" for more information.
adaDelta
  :: (Monad m, ParamSet p)
  => Config
    -- ^ AdaDelta configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element
  -> SGD m e p
adaDelta Config{..} gradient net0 =

  let zr = zero net0 
   in go (0 :: Integer) zr zr zr net0

  where

    go k expSqGradPrev expSqDeltaPrev deltaPrev net = do
      x <- P.await
      let grad = gradient x net
          expSqGrad = scale decay expSqGradPrev
                `add` scale (1-decay) (square grad)
          rmsGrad = squareRoot (pmap (+eps) expSqGrad)
          expSqDelta = scale decay expSqDeltaPrev
                 `add` scale (1-decay) (square deltaPrev)
          rmsDelta = squareRoot (pmap (+eps) expSqDelta)
          delta = (rmsDelta `mul` grad) `div` rmsGrad
          newNet = net `sub` delta
      P.yield newNet
      go (k+1) expSqGrad expSqDelta delta newNet


-------------------------------
-- Utils
-------------------------------


-- | Scaling
scale :: ParamSet p => Double -> p -> p
scale x = pmap (*x)
{-# INLINE scale #-}


-- | Root square
squareRoot :: ParamSet p => p -> p
squareRoot = pmap sqrt
{-# INLINE squareRoot #-}


-- | Square
square :: ParamSet p => p -> p
square x = x `mul` x
{-# INLINE square #-}
