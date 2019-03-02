{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | High-level stochastic gradient descent (SGD) module.
--
-- SGD can be used to find the optimal parameter values, minimizing a given
-- objective function.  This module requires that (i) the parameters have a
-- `ParamSet` instance, and (ii) the gradient of the objective function is
-- known (you can determine it automatically using, e.g., the backprop
-- library).


module Numeric.SGD
  (
  -- * Pure SGD
    runSgd

  -- * IO-based SGD
  , Config (..)
  , def
  , runSgdIO

  -- * Combinators
  , pipeSeq
  , pipeRan
  , result
  , every

  -- * Re-exports
  , module Numeric.SGD.ParamSet
  , module Numeric.SGD.DataSet
  ) where


import           GHC.Generics (Generic)
import           Numeric.Natural (Natural)

import qualified System.Random as R

import           Control.Monad (when, forM_)

import           Data.Functor.Identity (Identity(..))
import qualified Data.IORef as IO
import           Data.Default

import qualified Pipes as P
import qualified Pipes.Prelude as P
import           Pipes ((>->))

-- import qualified Numeric.SGD.AdaDelta as Ada
-- import qualified Numeric.SGD.Momentum as Mom
import           Numeric.SGD.ParamSet
import           Numeric.SGD.DataSet


------------------------------- 
-- Config
-------------------------------


-- -- | Available SGD methods, together with the corresponding configurations
-- data Method
--   = AdaDelta Ada.Config
--   | Momentum Mom.Config
--   deriving (Show, Eq, Ord, Generic)


-------------------------------
-- Pure SGD
-------------------------------


-- | Pure SGD method
type SGD m e p = p -> P.Pipe e p m ()


-- | Stochastic gradient descent, pure and simple
runSgd
  :: (ParamSet p)
  => SGD Identity e p
    -- ^ Selected SGD method
  -> [e]
    -- ^ Training data stream
  -> p
    -- ^ Initial parameters
  -> p
runSgd sgd dataSet p0 = runIdentity $
  result p0 
    (P.each dataSet >-> sgd p0)


------------------------------- 
-- Higher-level SGD
-------------------------------


-- | High-level IO-based SGD configuration
data Config = Config
  { iterNum :: Natural
    -- ^ Number of iteration over the entire training dataset
  , batchRandom :: Bool
    -- ^ Should the mini-batch be selected at random?  If not, the subsequent
    -- training elements will be picked sequentially.  Random selection gives
    -- no guarantee of seeing each training sample in every epoch.
  , reportEvery :: Double
    -- ^ How often the quality should be reported (with `1` meaning once per
    -- pass over the training data)
  } deriving (Show, Eq, Ord, Generic)

instance Default Config where
  def = Config
    { iterNum = 100
    , batchRandom = False
    , reportEvery = 1.0
    }


-- | Higher-level, IO-embedded stochastic gradient descent, which should be
-- enough to train models on large datasets.
--
-- An alternative is to use the simpler `sgd`, or to build a custom SGD
-- pipeline based on lower-level combinators (`pipeSeq`, `Ada.adaDelta`,
-- `every`, `result`, etc.).
runSgdIO
  :: (ParamSet p)
  => Config
    -- ^ SGD configuration
  -> SGD IO e p
    -- ^ Selected SGD method
  -> (e -> p -> Double)
    -- ^ Value of the objective function on a sample element (needed for model
    -- quality reporting)
  -> DataSet e
    -- ^ Training dataset
  -> p
    -- ^ Initial parameter values
  -> IO p
runSgdIO Config{..} sgd quality0 dataSet net0 = do
  report net0
  result net0 $ pipeSeq dataSet
    >-> sgd net0
    >-> P.take realIterNum
    >-> every realReportPeriod report
  where
    -- Iteration scaling
    iterScale x = fromIntegral (size dataSet) * x
    -- Number of iterations and reporting period
    realIterNum = ceiling $ iterScale (fromIntegral iterNum :: Double)
    realReportPeriod = ceiling $ iterScale reportEvery
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
-- Lower-level combinators
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


-- | Apply the given function every `k` param sets flowing downstream.
every :: (Monad m) => Int -> (p -> m ()) -> P.Pipe p p m x
every k f = do
  go (1 `mod` k)
  where
    go i = do
      paramSet <- P.await
      when (i == 0) $ do
        P.lift $ f paramSet
      P.yield paramSet
      go $ (i+1) `mod` k
