{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}

-- | Implementation of the AdaDelta algorithm as described in the following
-- paper:
--
--     * https://arxiv.org/pdf/1212.5701.pdf


module Numeric.SGD.AdaDelta
  ( Config(..)
  , adaDeltaM
  , adaDelta
  ) where


import           GHC.Generics (Generic)

import           Prelude hiding (div)
import           Control.Monad (when)

import qualified Pipes as P

import           Numeric.SGD.ParamSet
import           Numeric.SGD.Args


-- | AdaDelta configuration
data Config = Config
  { decay :: Double
    -- ^ Exponential decay parameter
  , eps   :: Double
    -- ^ Epsilon value
  } deriving (Show, Eq, Ord, Generic)


-- | Perform gradient descent using the AdaDelta algorithm.
adaDeltaM
  :: (Monad m, ParamSet p)
  => Config
    -- ^ AdaDelta configuration
  -> Args m p
    -- ^ General SGD arguments
  -> p 
    -- ^ Initial parameters
  -> m p
adaDeltaM Config{..} Args{..} net0 =

  let zr = zero net0 
   in go 0 zr zr zr net0

  where

    go k expSqGradPrev expSqDeltaPrev deltaPrev net
      | k > iterNum = return net
      | otherwise = do
          -- let netSize = norm_2 net
          when (k `mod` reportPeriod == 0) $ do
            report net
          -- grad <- netSize `seq` gradient net
          grad <- gradient net
          let expSqGrad = scale decay expSqGradPrev
                    `add` scale (1-decay) (square grad)
              rmsGrad = squareRoot (pmap (+eps) expSqGrad)
              expSqDelta = scale decay expSqDeltaPrev
                     `add` scale (1-decay) (square deltaPrev)
              rmsDelta = squareRoot (pmap (+eps) expSqDelta)
              delta = (rmsDelta `mul` grad) `div` rmsGrad
              -- delta = scale 0.01 grad `div` rmsGrad
              newNet = net `sub` delta
          go (k+1) expSqGrad expSqDelta delta newNet


-- | Perform gradient descent using the AdaDelta algorithm.
adaDelta
  :: (Monad m, ParamSet p)
  => Config
    -- ^ AdaDelta configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element.
  -> p 
    -- ^ Initial parameters
  -> P.Pipe e p m ()
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
