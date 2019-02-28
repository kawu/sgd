{-# LANGUAGE RecordWildCards #-}


module Numeric.SGD.AdaDelta
  ( Config(..)
  , adaDelta
  ) where


import           Prelude hiding (div)
import           Control.Monad (when)

import           Numeric.SGD.ParamSet


-- | AdaDelta configuration
data Config = Config
  { iterNum :: Int
    -- ^ Number of iteration to perform
  , reportEvery :: Int
    -- ^ How often (in terms of the numer of iterations) to report the quality
  , decay :: Double
    -- ^ Exponential decay parameter
  , eps   :: Double
    -- ^ Epsilon value
  }


-- -- | AdaDelta ,,dynamic'' configuration
-- data Dyna p = Dyna
--   { gradient :: p -> IO p
--     -- ^ Gradient on (a part of) the training data w.r.t. the given parameter
--     -- set.  Embedded in the IO monad because of the stochasticity of the
--     -- process.
--   , quality :: p -> IO Double
--     -- ^ Quality measure.  Embedded in the IO monad for convenience.  You
--     -- may, for instance, pick a random subset of the training dataset to
--     -- calculate the quality.
--   }


-- | Perform gradient descent using the AdaDelta algorithm.
adaDelta
  :: (ParamSet p)
  => Config
    -- ^ AdaDelta configuration
  -> (p -> IO p)
    -- ^ Gradient on (some part of) the training data w.r.t. the given
    -- parameter set.  Embedded in the IO monad because of the stochasticity of
    -- the process.
  -> (p -> IO Double)
    -- ^ Quality measure.  Embedded in the IO monad for convenience.  You
    -- may, for instance, pick a random subset of the training dataset to
    -- calculate the quality.
  -> p 
    -- ^ Initial parameters
  -> IO p
adaDelta Config{..} gradient quality net0 =

  let zr = zero net0 
   in go 0 zr zr zr net0

  where

    go k expSqGradPrev expSqDeltaPrev deltaPrev net
      | k > iterNum = return net
      | otherwise = do
          let netSize = norm_2 net
          when (k `mod` reportEvery == 0) $ do
            putStr . show =<< quality net
            putStrLn $ " (norm_2 = " ++ show netSize ++ ")"
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
