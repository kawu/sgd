{-# LANGUAGE RecordWildCards #-}


module Numeric.SGD.AdaDelta
  ( Config(..)
  , Dyna(..)
  , adaDelta
  ) where


import           Prelude hiding (div)
import           Control.Monad (when)

import           Numeric.SGD.ParamSet


-- | AdaDelta configuration
data Config = Config
  { decay :: Double
    -- ^ Exponential decay parameter
  , eps   :: Double
    -- ^ Epsilon value
  }


-- | AdaDelta ,,dynamic'' configuration
data Dyna p = Dyna
  { gradient :: p -> IO p
    -- ^ Gradient on (a part of) the training data w.r.t. the given parameter
    -- set.  Embedded in the IO monad because of the stochasticity of the
    -- process.
  , quality :: p -> IO Double
    -- ^ Quality measure.  Embedded in the IO monad for convenience.  You
    -- may, for instance, pick a random subset of the training dataset to
    -- calculate the quality.
  , iterNum :: Int
    -- ^ Number of iteration to perform
  , reportEvery :: Int
    -- ^ How often (in terms of the numer of iterations) to report the quality
  }


-- class ParamSet p where
--   -- | Zero
--   zero :: p
--   -- | Mapping
--   pmap :: (Double -> Double) -> p -> p
-- 
--   -- | Negation
--   neg :: p -> p
--   neg = pmap (\x -> -x)
--   -- | Addition
--   add :: p -> p -> p
--   add x y = x `sub` neg y
--   -- | Substruction
--   sub :: p -> p -> p
--   sub x y = x `add` neg y
-- 
--   -- | Element-wise multiplication
--   mul :: p -> p -> p
--   mul x y = x `div` pmap (1.0/) y
--   -- | Element-wise division
--   div :: p -> p -> p
--   div x y = x `mul` pmap (1.0/) y


-- | Perform gradient descent using the AdaDelta algorithm.
adaDelta
  :: (ParamSet p)
  => Config
    -- ^ AdaDelta configuration
  -> Dyna p
    -- ^ AdaDelta ,,dynamic'' configuration
  -> p 
    -- ^ Initial parameters
  -> IO p
adaDelta Config{..} Dyna{..} net0 =

  go 0 zero zero zero net0

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
