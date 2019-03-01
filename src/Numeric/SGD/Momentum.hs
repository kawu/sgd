{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Implementation of SGD with momentum


module Numeric.SGD.Momentum
  ( Config(..)
  , Args(..)
  , momentumM
  ) where


import           GHC.Generics (Generic)

import           Control.Monad (when)

import           Numeric.SGD.ParamSet
import           Numeric.SGD.Args


-- -- | AdaDelta configuration
-- data Config = Config
--   { iterNum :: Int
--     -- ^ Number of iterations to perform
--   , reportEvery :: Int
--     -- ^ The quality will be reported every `reportEvery` iterations
--   , gain0 :: Double
--   -- ^ Initial gain parameter
--   , tau :: Double
--   -- ^ After how many gradient calculations the gain parameter is halved
--   , gamma :: Double
--   -- ^ Exponentional decay parameter
--   }
-- 
-- 
-- -- | Perform simple gradient descent with momentum.
-- momentum 
--   :: (ParamSet p)
--   => Config
--     -- ^ Momentum configuration
--   -> (p -> IO p)
--     -- ^ Gradient on (some part of) the training data w.r.t. the given
--     -- parameter set.  Embedded in the IO monad because of the stochasticity of
--     -- the process.
--   -> (p -> IO Double)
--     -- ^ Quality measure.  Embedded in the IO monad for convenience.  You
--     -- may, for instance, pick a random subset of the training dataset to
--     -- calculate the quality.
--   -> p 
--     -- ^ Initial parameters
--   -> IO p
-- momentum Config{..} gradient quality net0 =
-- 
--   go 0 (zero net0) net0
-- 
--   where
-- 
--     -- Gain in the k-th iteration
--     gain k
--       = (gain0 * tau)
--       / (tau + fromIntegral k)
-- 
--     go k moment net
--       | k > iterNum = return net
--       | otherwise = do
--           let netSize = norm_2 net
--           when (k `mod` reportEvery == 0) $ do
--             putStr . show =<< quality net
--             putStrLn $ " (norm_2 = " ++ show netSize ++ ")"
--           grad <- scale (gain k) <$> gradient net
--           let moment' = scale gamma moment `add` grad
--               newNet = net `sub` moment'
--           go (k+1) moment' newNet


-------------------------------
-- New version
-------------------------------


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


-------------------------------
-- Utils
-------------------------------


-- | Scaling
scale :: ParamSet p => Double -> p -> p
scale x = pmap (*x)
{-# INLINE scale #-}
