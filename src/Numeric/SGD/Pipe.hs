{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Pipe-based interface


module Numeric.SGD.Pipe
  ( Config(..)
  , Method(..)
  , sgd
  , result
  , every
  , pipeSeq
  , pipeRan
  ) where


import           GHC.Generics (Generic)
import           Numeric.Natural (Natural)

import           Control.Monad (when, forM_)

import qualified System.Random as R

import qualified Data.IORef as IO

import qualified Pipes as P
import qualified Pipes.Prelude as P
import           Pipes ((>->))

import           Numeric.SGD.DataSet
import           Numeric.SGD.ParamSet
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Momentum as Mom


------------------------------- 
-- Data
-------------------------------


-- | Top-level SGD configuration
data Config = Config
  { iterNum :: Natural
    -- ^ Number of iteration over the entire training dataset
  , batchRandom :: Bool
    -- ^ Should the mini-batch be selected at random?  If not, the subsequent
    -- training elements will be picked sequentially.  Random selection gives
    -- no guarantee of seeing each training sample in every epoch.  Use `False`
    -- if unsure.
  , method :: Method
    -- ^ Selected SGD method
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


-- | Pipe-based SGD.
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
  let sgdPipe =
        case method of
          Momentum cfg -> Mom.momentum cfg
            -- (cfg {Mom.tau = iterScale (Mom.tau cfg)})
            grad0 net0
          AdaDelta cfg -> Ada.adaDelta cfg grad0 net0
  report net0
  result net0 $ pipeSeq dataSet
    >-> sgdPipe
    >-> P.take realIterNum
    >-> every realReportPeriod report
  where
    -- Iteration scaling
    iterScale x = fromIntegral (size dataSet) * x
    -- Number of iterations and reporting period
    realIterNum = ceiling $ iterScale (fromIntegral iterNum :: Double)
    realReportPeriod = ceiling $ iterScale reportPeriod
    -- Network quality over the entire training dataset
    report net = do
      putStr . show =<< quality net
      putStrLn $ " (norm_2 = " ++ show (norm_2 net) ++ ")"
    quality net = do
      res <- IO.newIORef 0.0
      forM_ [0 .. size dataSet - 1] $ \ix -> do
        x <- elemAt dataSet ix
        IO.modifyIORef' res (+ quality0 x net)
      IO.readIORef res


------------------------------- 
-- Dataset producers
-------------------------------


-- | Pipe the dataset sequentially in a loop.
pipeSeq :: DataSet e -> P.Producer e IO ()
pipeSeq dataSet = do
  go (0 :: Int)
  where
    go k
      | k >= size dataSet = go 0
      | otherwise = do
          x <- P.lift $ elemAt dataSet k
          P.yield x
          go (k+1)


-- | Pipe the dataset randomly in a loop.
pipeRan :: DataSet e -> P.Producer e IO ()
pipeRan dataSet = do
  x <- P.lift $ do
    ix <- R.randomRIO (0, size dataSet - 1)
    elemAt dataSet ix
  P.yield x
  pipeRan dataSet


------------------------------- 
-- Utils
-------------------------------


-- | Extract the result of the SGD calculation (the last parameter
-- set flowing downstream).
result
  :: (Monad m)
  => p     
    -- ^ Default value (in case the stream is empty)
  -> P.Producer p m ()
    -- ^ Stream of parameter sets
  -> m p
result pDef = fmap (maybe pDef id) . P.last


-- | Report every `k`-th parameter set flowing downstream.
every :: (Monad m) => Int -> (p -> m ()) -> P.Pipe p p m x
every k report = do
  go (1 `mod` k)
  where
    go i = do
      paramSet <- P.await
      when (i == 0) $ do
        P.lift $ report paramSet
      P.yield paramSet
      go $ (i+1) `mod` k
