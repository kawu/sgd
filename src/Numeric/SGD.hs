{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Various stochastic gradient descent (SGD) optimization methods.


module Numeric.SGD
  ( Config (..)
  , Method (..)
  , sgd
  , module Numeric.SGD.ParamSet
  , module Numeric.SGD.DataSet
  ) where


import           GHC.Generics (Generic)
import           Numeric.Natural (Natural)

import           Prelude hiding (elem)
import           Control.Monad (forM_)

import           Data.List (foldl')
import qualified Data.IORef as IO

import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Momentum as Mom
import           Numeric.SGD.ParamSet
import           Numeric.SGD.DataSet
import qualified Numeric.SGD.Args as Args


------------------------------- 
-- Data
-------------------------------


-- | Top-level SGD configuration
data Config = Config
  { iterNum :: Natural
    -- ^ Number of iteration over the entire training dataset
  , batchSize :: Natural
    -- ^ Size of the SGD batch (use `1` if unsure)
  , batchRandom :: Bool
    -- ^ Should the batch be selected at random?  If not, the subsequent
    -- batches will be picked sequentially.  Random batch selection gives no
    -- guarantee of seeing each training sample in every epoch.  Use `False` if
    -- unsure.
  , method :: Method
    -- ^ Selected SGD method (`AdaDelta` seems like a safe default)
  , reportPeriod :: Double
    -- ^ How often the quality should be reported (with `1` meaning once per
    -- pass over the training data)
  } deriving (Show, Eq, Ord, Generic)


-- | Different SGD methods, together with the corresponding configurations
data Method
  = AdaDelta {adaDeltaCfg :: Ada.Config}
  | Momentum {momentumCfg :: Mom.Config}
  deriving (Show, Eq, Ord, Generic)


------------------------------- 
-- SGD
-------------------------------


-- | Perform SGD.
sgd
  :: (ParamSet p)
  => Config
  -> DataSet e
  -> (e -> p -> p)
    -- ^ Network gradient on a sample element
  -> (e -> p -> Double)
    -- ^ Value of the objective function on a sample element
  -> p
    -- ^ Initial parameter values
  -> IO p
sgd Config{..} dataSet grad0 quality0 net0 = do
  grad <-
    if batchRandom
       then pure randomGrad
       else do
         ixRef <- IO.newIORef 0 
         pure (determGrad ixRef)
  case method of
    AdaDelta cfg ->
      Ada.adaDeltaM cfg (args grad) net0
    Momentum cfg ->
      Mom.momentumM
        (cfg {Mom.tau = iterScale (Mom.tau cfg)})
        (args grad) net0
  where
    -- General SGD arguments
    args grad = Args.Args
      { Args.iterNum = realIterNum
      , Args.gradient = grad
      , Args.reportPeriod = realReportPeriod
      , Args.report = report
      }
    -- Iteration scaling
    iterScale x
      = fromIntegral (size dataSet) * x
      / (fromIntegral batchSize :: Double)
    -- Number of iterations and reporting period
    realIterNum = ceiling $ iterScale (fromIntegral iterNum)
    realReportPeriod = ceiling $ iterScale reportPeriod
    -- Gradient over a random sample 
    randomGrad net = do
      sample <- randomSample (fromIntegral batchSize) dataSet
      return $ gradSample sample net
    -- Gradient over a deterministically selected sample
    determGrad ixRef net = do
      p <- IO.readIORef ixRef
      let q = p + fromIntegral batchSize - 1
      sample <- range p q dataSet
      IO.writeIORef ixRef $ p + 1 `mod` size dataSet
      return $ gradSample sample net
    -- Gradient on a training sample
    gradSample sample net = do
      let grads = map (\x -> grad0 x net) sample
       in foldl' add (zero net) grads
    -- Network quality over the entire training dataset
    report net = do
      putStr . show =<< quality net
      putStrLn $ " (norm_2 = " ++ show (norm_2 net) ++ ")"
    quality net = do
      res <- IO.newIORef 0.0
      forM_ [0 .. size dataSet - 1] $ \ix -> do
        elem <- elemAt dataSet ix
        IO.modifyIORef' res (+ quality0 elem net)
      IO.readIORef res


-- | Dataset sample over a given (inclusive) range
range :: Int -> Int -> DataSet a -> IO [a]
range p q dataSet
  | p > q = return []
  | otherwise = do
      x <- elemAt dataSet (p `mod` size dataSet)
      (x:) <$> range (p+1) q dataSet
