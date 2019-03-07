{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Main module of the stochastic gradient descent (SGD) library. 
--
-- SGD is a method for optimizing a global objective function defined as a sum
-- of smaller, differentiable functions.  The individual component functions
-- share the same set of parameters, represented by the `ParamSet` class.
--
-- To perform SGD, the gradients of the individual functions need to be
-- determined.  This can be done manually or automatically, using one of the
-- automatic differentiation libraries (ad, backprop) available in Haskell.
--
-- For instance, let's say we have a list of functions defined as:
--
-- > funs = [\x -> 0.3*x^2, \x -> -2*x, const 3, sin]
--
-- The global objective is then defined as:
--
-- > objective x = sum $ map ($x) funs
--
-- We can manually determine the individual derivatives:
--
-- > derivs = [\x -> 0.6*x, const (-2), const 0, cos]
--
-- or use an automatic differentiation library, for instance:
--
-- > import qualified Numeric.AD as AD
-- > derivs = map
-- >   (\k -> AD.diff (funs !! k))
-- >   [0..length funs-1]
--
-- Finally, `run` allows to approach a (potentially local) minimum of the
-- global objective function:
--
-- >>> run (momentum def id) (take 10000 $ cycle derivs) 0.0
-- 4.180177042912455
--
-- where:
-- 
--     * @(take 10000 $ cycle derivs)@ is the stream of training examples
--     * @(momentum def id)@ is the selected SGD variant (`Mom.momentum`),
--     supplied with the default configuration (`def`) and the function (`id`)
--     for calculating the gradient from a training example
--     * @0.0@ is the initial parameter value


module Numeric.SGD
  (
  -- * SGD variants
    Mom.momentum
  , Ada.adaDelta
  , Adam.adam

  -- * Pure SGD
  , run

  -- * IO-based SGD
  , Config (..)
  , runIO

  -- * Combinators
  , pipeSeq
  , pipeRan
  , batch
  , batchGrad
  , result
  , every

  -- * Re-exports
  , def
  ) where


import           GHC.Generics (Generic)
-- import           GHC.Conc (numCapabilities)
import           Numeric.Natural (Natural)

-- import qualified System.Random as R

import           Control.Monad (when, forM_, forever)
import           Control.Parallel.Strategies (rseq, parMap)
import qualified Control.Monad.State.Strict as State

import           Data.Functor.Identity (Identity(..))
import           Data.List (foldl1') -- , transpose)

import qualified Data.IORef as IO
import           Data.Default

import qualified Pipes as P
import qualified Pipes.Prelude as P
import           Pipes ((>->))

import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Adam as Adam
import           Numeric.SGD.Type
import           Numeric.SGD.ParamSet
import           Numeric.SGD.DataSet


-------------------------------
-- Pure SGD
-------------------------------


-- | Traverse all the elements in the training data stream in one pass,
-- calculate the subsequent gradients, and apply them progressively starting
-- from the initial parameter values.
--
-- Consider using `runIO` if your training dataset is large.
run
  :: (ParamSet p)
  => SGD Identity e p
    -- ^ Selected SGD method
  -> [e]
    -- ^ Training data stream
  -> p
    -- ^ Initial parameters
  -> p
run sgd dataSet p0 = runIdentity $
  result p0 
    (P.each dataSet >-> sgd p0)


------------------------------- 
-- Higher-level SGD
-------------------------------


-- | High-level IO-based SGD configuration
data Config = Config
  { iterNum :: Natural
    -- ^ Number of iteration over the entire training dataset
  , batchSize :: Natural
    -- ^ Mini-batch size
  , batchRandom :: Bool
    -- ^ Should the mini-batch be selected at random?  If not, the subsequent
    -- training elements will be picked sequentially.  Random selection gives
    -- no guarantee of seeing each training sample in every epoch.
  , reportEvery :: Double
    -- ^ How often the value of the objective function should be reported (with
    -- @1@ meaning once per pass over the training data)
  } deriving (Show, Eq, Ord, Generic)

instance Default Config where
  def = Config
    { iterNum = 100
    , batchSize = 1
    , batchRandom = False
    , reportEvery = 1.0
    }


-- | Perform SGD in the IO monad, regularly reporting the value of the
-- objective function on the entire dataset.  A higher-level wrapper which
-- should be convenient to use when the training dataset is large.
--
-- An alternative is to use the simpler function `run`, or to build a custom
-- SGD pipeline based on lower-level combinators (`pipeSeq`, `Ada.adaDelta`,
-- `every`, `result`, etc.).
runIO
  :: (ParamSet p)
  => Config
    -- ^ SGD configuration
  -> SGD IO [e] p
    -- ^ SGD pipe consuming mini-batches of dataset elements
  -> (e -> p -> Double)
    -- ^ Value of the objective function on a dataset element (used for model
    -- quality reporting)
  -> DataSet e
    -- ^ Training dataset
  -> p
    -- ^ Initial parameter values
  -> IO p
runIO Config{..} sgd quality0 dataSet net0 = do
  report net0
  result net0 $ pipeData dataSet
    >-> batch (fromIntegral batchSize)
    >-> sgd net0
    >-> P.take realIterNum
    >-> every realReportPeriod report
  where
    pipeData = forever .
      if batchRandom
         then pipeRan
         else pipeSeq
    -- Iteration (epoch) scaling
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


-- | Pipe all the elements in the dataset sequentially.
pipeSeq :: DataSet e -> P.Producer e IO ()
pipeSeq dataSet = do
  go (0 :: Int)
  where
    go k
      | k >= size dataSet = return ()
      | otherwise = do
          x <- P.lift $ elemAt dataSet k
          P.yield x
          go (k+1)


-- | Pipe all the elements in the dataset in a random order.
pipeRan :: DataSet e -> P.Producer e IO ()
pipeRan dataSet0 = do
  dataSet <- P.lift $ shuffle dataSet0
  pipeSeq dataSet


-- | Group dataset elements into (mini-)batches of the given size.
batch :: (Monad m) => Int -> P.Pipe e [e] m ()
batch k = flip State.evalStateT [] . forever $ do
  x <- P.lift P.await
  xs <- State.get
  let xs' = take k (x:xs)
  when (length xs' == k) $ do
    P.lift (P.yield xs')
  State.put xs'


-- | Adapt the gradient function to handle (mini-)batches.
-- TODO: Mention that `p` has to be strict for parallelism?
batchGrad
  :: (ParamSet p)
  => (e -> p -> p)
  -> ([e] -> p -> p)
batchGrad grad xs param =
--   = foldl1' add
--   . parMap rseq gradOn
--   $ partition numCapabilities xs
--   where
--     gradOn = foldl1' add . map (flip grad param)
  case parMap rseq (\e -> grad e param) xs of
    [] -> param
    ps -> foldl1' add ps


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


-- | Apply the given function every @k@ param sets flowing downstream.
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


-------------------------------
-- Utils
-------------------------------


-- partition :: Int -> [a] -> [[a]]
-- partition n =
--     transpose . group n
--   where
--     group _ [] = []
--     group k xs = take k xs : (group k $ drop k xs)
