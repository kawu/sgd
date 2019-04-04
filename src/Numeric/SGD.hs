{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Main module of the stochastic gradient descent (SGD) library. 
--
-- SGD is a method for optimizing a global objective function defined as a sum
-- of smaller, differentiable functions.  The individual component functions
-- share the same set of parameters, represented by the `ParamSet` class.  This
-- allows for heterogeneous parameter representation (vectors, maps, custom
-- records, etc.).
--
-- The library adopts a `P.Pipe`-based interface in which `SGD` takes the form
-- of a process consuming dataset subsets (the so-called mini-batches) and
-- producing a stream of output parameter values.  The library implements
-- different variants of `SGD` (`Mom.momentum`, `Adam.adam`, `Ada.adaDelta`)
-- which can be executed in either the pure context (`run`) or in IO (`runIO`).
-- The use of lower-level pipe-processing combinators (`pipeRan`, `batch`,
-- `result`, etc.) is also possible.
--
-- To perform SGD, the gradients of the individual functions need to be
-- determined.  This can be done manually or automatically, using an automatic
-- differentiation library (<http://hackage.haskell.org/package/ad ad>,
-- <http://hackage.haskell.org/package/backprop backprop>).
--

module Numeric.SGD
  (
  -- * Example
  -- $example

  -- * SGD variants
    Mom.momentum
  , Ada.adaDelta
  , Adam.adam

  -- * Pure SGD
  , run

  -- * IO-based SGD
  , Config (..)
  , iterNumPerEpoch
  , reportObjective
  , objectiveWith
  , runIO

  -- * Combinators
  -- ** Input
  , pipeSeq
  , pipeRan
  -- ** Batch
  , batch
  , batchGradSeq
  , batchGradPar
  , batchGradPar'
  -- ** Output
  , result
  -- ** Misc
  , keepEvery
  , decreasingBy

  -- * Re-exports
  , def
  ) where


import           GHC.Generics (Generic)
-- import           GHC.Conc (numCapabilities)
import           Numeric.Natural (Natural)

-- import qualified System.Random as R

import           Control.Monad (when, forM_, forever)
import           Control.Parallel.Strategies (parMap, rseq, rdeepseq, Strategy)
import           Control.DeepSeq (NFData)
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


{- $example

  Let's say we have a list of functions defined as:

> funs = [\x -> 0.3*x^2, \x -> -2*x, const 3, sin]

  The global objective (which we want to minimize) is then defined as:

> objective x = sum $ map ($x) funs

  To perform SGD, we can either manually determine the individual derivatives:

> derivs = [\x -> 0.6*x, const (-2), const 0, cos]

  or use an automatic differentiation library, for instance:

> import qualified Numeric.AD as AD
> derivs = map
>   (\k -> AD.diff (funs !! k))
>   [0..length funs-1]

  Finally, `run` allows to approach a (potentially local) minimum of the
  global objective function:

>>> run (momentum def id) (take 10000 $ cycle derivs) 0.0
4.180177042912455

  where:

    * @(take 10000 $ cycle derivs)@ is the stream of training examples
    * @(momentum def id)@ is the selected SGD variant (`Mom.momentum`),
    supplied with the default configuration (`def`) and the function (`id`)
    for calculating the gradient from a training example
    * @0.0@ is the initial parameter value

-}


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
  , batchOverlap :: Natural
    -- ^ The number of overlapping elements in subsequent mini-batches
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
    , batchOverlap = 0
    , batchRandom = False
    , reportEvery = 1.0
    }


-- | Number of new elements in each new batch
batchNew :: Config -> Int
batchNew cfg = max 1
  ( fromIntegral (batchSize cfg)
  - fromIntegral (batchOverlap cfg)
  )


-- | Calculate the effective number of SGD iterations (and gradient
-- calculations) performed per epoch.
iterNumPerEpoch
  :: (Integral a)
  => Config
  -> a -- ^ Dataset size
  -> Double
iterNumPerEpoch cfg size =
  fromIntegral size / fromIntegral (batchNew cfg)


-- | Report the total objective value on stdout.
reportObjective
  :: (ParamSet p)
  => (e -> p -> Double)
    -- ^ Value of the objective function on a dataset element
  -> DataSet e
    -- ^ Training dataset
  -> p -> IO Double
reportObjective objAt dataSet net = do
  q <- objectiveWith objAt dataSet net
  putStr $ show q
  putStrLn $ " (norm_2 = " ++ show (norm_2 net) ++ ")"
  return q


-- | Value of the objective function over the entire dataset (i.e. the sum of
-- the objectives on all dataset elements).
objectiveWith
  :: (e -> p -> Double)
    -- ^ Value of the objective function on a dataset element
  -> DataSet e
    -- ^ Training dataset
  -> p -> IO Double
objectiveWith objAt dataSet net = do
  res <- IO.newIORef 0.0
  forM_ [0 .. size dataSet - 1] $ \ix -> do
    x <- elemAt dataSet ix
    IO.modifyIORef' res (+ objAt x net)
  IO.readIORef res


-- | Perform SGD in the IO monad, regularly reporting the value of the
-- objective function on the entire dataset.  A higher-level wrapper which
-- should be convenient to use when the training dataset is large.
--
-- An alternative is to use the simpler function `run`, or to build a custom
-- SGD pipeline based on lower-level combinators (`pipeSeq`, `batch`,
-- `Adam.adam`, `keepEvery`, `result`, etc.).
runIO
  :: (ParamSet p)
  => Config
    -- ^ SGD configuration
  -> SGD IO [e] p
    -- ^ SGD pipe consuming mini-batches of dataset elements
  -> (p -> IO Double)
    -- ^ Quality reporting function (the reporting frequency is specified
    -- via `reportEvery`)
  -> DataSet e
    -- ^ Training dataset
  -> p
    -- ^ Initial parameter values
  -> IO p
runIO cfg@Config{..} sgd reportObj dataSet net0 = do
  _ <- reportObj net0
  result net0 $ pipeData dataSet
    >-> batch (fromIntegral batchSize)
    >-> batchFilter
    >-> sgd net0
    >-> keepEvery realReportPeriod
    >-> P.take (fromIntegral iterNum)
    >-> decreasingBy reportObj
  where
    -- Data streaming function
    pipeData = forever .
      if batchRandom
         then pipeRan
         else pipeSeq
    -- Batch stream filter
    batchFilter = do
      P.await >>= P.yield
      keepEvery (batchNew cfg)
    -- Iteration (epoch) scaling
    realReportPeriod = ceiling $
      reportEvery * iterNumPerEpoch cfg (size dataSet)


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


-- | Adapt the gradient function to handle (mini-)batches.  Relies on the @p@'s
-- `NFData` instance to efficiently calculate gradients in parallel.
batchGradPar
  :: (ParamSet p, NFData p)
  => (e -> p -> p)
  -> ([e] -> p -> p)
batchGradPar = batchGradWith rdeepseq


-- | A version of `batchGradPar` with no `NFData` constraint.  Evaluates the
-- sub-gradients calculated in parallel to weak head normal form.
batchGradPar'
  :: (ParamSet p)
  => (e -> p -> p)
  -> ([e] -> p -> p)
batchGradPar' = batchGradWith rseq


-- | Adapt the gradient function to handle (mini-)batches.  The sub-gradients
-- of the individual batch elements are evaluated in parallel based on the
-- given `Strategy`.
batchGradWith
  :: (ParamSet p)
  => Strategy p
  -> (e -> p -> p)
  -> ([e] -> p -> p)
batchGradWith strategy grad xs param =
  case parMap strategy (\e -> grad e param) xs of
    [] -> param
    -- TODO: the fold is sequential, we could try to parallize it as well.
    ps -> foldl1' add ps


-- | Adapt the gradient function to handle (mini-)batches.  The function
-- calculates the individual sub-gradients sequentially.
batchGradSeq
  :: (ParamSet p)
  => (e -> p -> p)
  -> ([e] -> p -> p)
batchGradSeq grad xs param =
  case map (flip grad param) xs of
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


-- -- | Apply the given monadic function to every @k@-th value flowing downstream.
-- every :: (Monad m) => Int -> (p -> m ()) -> P.Pipe p p m x
-- every k f = do
--   go (1 `mod` k)
--   where
--     go i = do
--       paramSet <- P.await
--       when (i == 0) $ do
--         P.lift $ f paramSet
--       P.yield paramSet
--       go $ (i+1) `mod` k


-- | Keep every @k@-th element flowing downstream and discard all the others.
keepEvery :: (Monad m) => Int -> P.Pipe a a m x
keepEvery k = forever $ do
  sequence_ $ replicate (k-1) P.await
  P.await >>= P.yield
-- keepEvery k = do
--   go (1 `mod` k)
--   where
--     go i = do
--       x <- P.await
--       when (i == 0) $ do
--         P.yield x
--       go $ (i+1) `mod` k


-- -- | Keep the elements with the corresponding `True` in the argument list.
-- --
-- -- TODO: (=) or (==) in the following example?  And is this example correct?
-- -- @
-- -- keep (forever True) = P.id
-- -- @
-- keep :: (Monad m) => [Bool] -> P.Pipe a a m ()
-- keep [] = return ()
-- keep (b:bs) = do
--   x <- P.await
--   when b (P.yield x)
--   keep bs
-- 
-- 
-- -- | Create the mask to `keep` each @k@-th element flowing downstream.
-- every :: Int -> [Bool]
-- every k = cycle $ replicate (k-1) False ++ [True]


-- | Make the stream decreasing in the given (monadic) function by discarding
-- elements with values higher than those already seen.
decreasingBy :: (Monad m, Ord a) => (p -> m a) -> P.Pipe p p m x
decreasingBy f = do
  x <- P.await
  v <- P.lift (f x)
  P.yield x
  go v
  where
    go w = do
      x <- P.await
      v <- P.lift (f x)
      when (v < w) (P.yield x)
      go (min v w)


-------------------------------
-- Utils
-------------------------------


-- partition :: Int -> [a] -> [[a]]
-- partition n =
--     transpose . group n
--   where
--     group _ [] = []
--     group k xs = take k xs : (group k $ drop k xs)
