{-# LANGUAGE RecordWildCards #-}

-- | Stochastic gradient descent implementation using mutable
-- vectors for efficient update of the parameters vector.
-- A user is provided with the immutable version of parameters vector
-- so he is able to compute the gradient outside the IO/ST monad.
-- Currently only the Gaussian priors are implemented.
--
-- This is a preliminary version of the SGD library and API may change
-- in future versions.

module Numeric.SGD
( SgdArgs (..)
, sgdArgsDefault
, Dataset
, Para
, GradGen
-- , sgd
, sgdIO
) where

import Control.Monad (forM_)
import Control.Monad.ST (RealWorld)
import Control.Concurrent.Async (mapConcurrently)
import GHC.Conc (numCapabilities)
import Data.List (transpose)
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as UM

import Numeric.SGD.LogSigned (LogSigned, toNorm)
import Numeric.SGD.Grad
import qualified Numeric.SGD.Grad.Map as GM
import qualified Numeric.SGD.Grad.Vector as GV

-- | SGD parameters controlling the learning process.
data SgdArgs = SgdArgs
    { -- | Size of the batch
      batchSize :: Int
    -- | Regularization variance
    , regVar    :: Double
    -- | Number of iterations
    , iterNum   :: Double
    -- | Initial gain parameter
    , gain0     :: Double
    -- | After how many iterations over the entire dataset
    -- the gain parameter is halved
    , tau       :: Double }

-- | Default SGD parameter values.
sgdArgsDefault :: SgdArgs
sgdArgsDefault = SgdArgs
    { batchSize = 30
    , regVar    = 10
    , iterNum   = 10
    , gain0     = 1
    , tau       = 5 }

-- | Dataset with elements of x type.
type Dataset x  = V.Vector x

-- | Vector of parameters.
type Para       = U.Vector Double 

-- | Type synonym for mutable vector with Double values.
type MVect      = UM.MVector RealWorld Double

-- -- | Pure version of the stochastic gradient descent method.
-- sgd :: SgdArgs              -- ^ SGD parameter values
--     -> (Para -> x -> Grad)  -- ^ Gradient for dataset element
--     -> Dataset x            -- ^ Dataset
--     -> Para                 -- ^ Starting point
--     -> Para                 -- ^ SGD result
-- sgd sgdArgs mkGrad dataset x0 =
--     let dummy _ _ = return ()
--     in  runST $ sgdM sgdArgs dummy mkGrad dataset x0

-- | Gradient generator.  Just a type synonym for a list
-- of (position, number to add) pairs.
type GradGen = [(Int, LogSigned)]

-- | Monadic version of the stochastic gradient descent method.
-- A notification function can be used to provide user with
-- information about the progress of the learning.
sgdIO
    :: Bool                     -- ^ Use vector gradients
    -> SgdArgs                  -- ^ SGD parameter values
    -> (Para -> Int -> IO ())   -- ^ Notification run every update
    -> (Para -> x -> GradGen)   -- ^ Gradient generator on element
    -> Dataset x                -- ^ Dataset
    -> Para                     -- ^ Starting point
    -> IO Para                  -- ^ SGD result
sgdIO False sgdArgs notify mkGrad =
    let fillGrad :: GradGen -> GM.Grad -> IO GM.Grad
        fillGrad = fill
        fillWith para xs = fillGrad (concatMap (mkGrad para) xs)
    in  sgdIO' sgdArgs notify fillWith
sgdIO True sgdArgs notify mkGrad =
    let fillGrad :: GradGen -> GV.Grad -> IO GV.Grad
        fillGrad = fill
        fillWith para xs = fillGrad (concatMap (mkGrad para) xs)
    in  sgdIO' sgdArgs notify fillWith

-- | Monadic version of the stochastic gradient descent method.
-- A notification function can be used to provide user with
-- information about the progress of the learning.
sgdIO'
    :: GradIO v
    => SgdArgs                      -- ^ SGD parameter values
    -> (Para -> Int -> IO ())       -- ^ Notification run every update
    -> (Para -> [x] -> v -> IO v)   -- ^ Fill gradient given dataset elements
    -> Dataset x                    -- ^ Dataset
    -> Para                         -- ^ Starting point
    -> IO Para                      -- ^ SGD result
sgdIO' SgdArgs{..} notify mkGrad dataset x0 = do
    u   <- UM.new (U.length x0)
    vs  <- sequence . replicate numCapabilities $ empty (U.length x0)
    doIt u vs 0 (R.mkStdGen 0) =<< U.thaw x0
  where
    -- | Gain in k-th iteration.
    gain k = (gain0 * tau) / (tau + done k)
    -- | Number of completed iterations over the full dataset.
    done k
        = fromIntegral (k * batchSize)
        / fromIntegral (V.length dataset) 

    doIt u vs k stdGen x
      | done k > iterNum = do
        frozen <- U.unsafeFreeze x
        notify frozen k
        return frozen
      | otherwise = do
        let (batch, stdGen') = sample stdGen batchSize dataset
            parts = partition (length vs) batch

        -- Freeze mutable vector of parameters. The frozen version is
        -- then supplied to external mkGrad function provided by user.
        frozen <- U.unsafeFreeze x
        notify frozen k

        -- let grad = parUnions (map (mkGrad frozen) batch)
        vs' <- mapConcurrently
                (uncurry $ mkGrad frozen)
                (zip parts vs)
        grad <- unionsTo (tail vs') (head vs')

        addUp grad u
	_ <- mapConcurrently clear vs

        scale (gain k) u

        x' <- U.unsafeThaw frozen
        apply u x'
        doIt u vs (k+1) stdGen' x'

-- | Add up all gradients and store results in normal domain.
addUp :: GradIO a => a -> MVect -> IO ()
addUp grad v = do
    UM.set v 0
    xs <- content grad
    forM_ xs $ \(i, x) -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (toNorm x + y)

-- | Scale the vector by the given value.
scale :: Double -> MVect -> IO ()
scale c v = do
    forM_ [0 .. UM.length v - 1] $ \i -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (c * y)

-- | Apply gradient to the parameters vector, that is add the first vector to
-- the second one.
apply :: MVect -> MVect -> IO ()
apply w v = do 
    forM_ [0 .. UM.length v - 1] $ \i -> do
        x <- UM.unsafeRead v i
        y <- UM.unsafeRead w i
        UM.unsafeWrite v i (x + y)

sample :: R.RandomGen g => g -> Int -> Dataset x -> ([x], g)
sample g 0 _       = ([], g)
sample g n dataset =
    let (xs, g') = sample g (n-1) dataset
        (i, g'') = R.next g'
        x = dataset V.! (i `mod` V.length dataset)
    in  (x:xs, g'')

partition :: Int -> [a] -> [[a]]
partition n =
    transpose . group n
  where
    group _ [] = []
    group k xs = take k xs : (group k $ drop k xs)
