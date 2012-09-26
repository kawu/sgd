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
, sgd
, sgdM
, module Numeric.SGD.Grad
) where

import Control.Monad (forM_)
import Control.Monad.ST (ST, runST)
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as UM
import qualified Control.Monad.Primitive as Prim

import Numeric.SGD.Grad

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
type MVect m    = UM.MVector (Prim.PrimState m) Double

-- | Pure version of the stochastic gradient descent method.
sgd :: SgdArgs              -- ^ SGD parameter values
    -> (Para -> x -> Grad)  -- ^ Gradient for dataset element
    -> Dataset x            -- ^ Dataset
    -> Para                 -- ^ Starting point
    -> Para                 -- ^ SGD result
sgd sgdArgs mkGrad dataset x0 =
    let dummy _ _ = return ()
    in  runST $ sgdM sgdArgs dummy mkGrad dataset x0

-- | Monadic version of the stochastic gradient descent method.
-- A notification function can be used to provide user with
-- information about the progress of the learning.
{-# SPECIALIZE sgdM :: SgdArgs
                    -> (Para -> Int -> IO ())
                    -> (Para -> x -> Grad)
                    -> Dataset x -> Para -> IO Para #-}
{-# SPECIALIZE sgdM :: SgdArgs
                    -> (Para -> Int -> ST s ())
                    -> (Para -> x -> Grad)
                    -> Dataset x -> Para -> ST s Para #-}
sgdM
    :: (Prim.PrimMonad m)
    => SgdArgs              -- ^ SGD parameter values
    -> (Para -> Int -> m ())    -- ^ Notification run every update
    -> (Para -> x -> Grad)  -- ^ Gradient for dataset element
    -> Dataset x            -- ^ Dataset
    -> Para                 -- ^ Starting point
    -> m Para               -- ^ SGD result
sgdM SgdArgs{..} notify mkGrad dataset x0 = do
    u <- UM.new (U.length x0)
    doIt u 0 (R.mkStdGen 0) =<< U.thaw x0
  where
    -- | Gain in k-th iteration.
    gain k = (gain0 * tau) / (tau + done k)
    -- | Number of completed iterations over the full dataset.
    done k
        = fromIntegral (k * batchSize)
        / fromIntegral (V.length dataset) 

    doIt u k stdGen x
      | done k > iterNum = do
        frozen <- U.unsafeFreeze x
        notify frozen k
        return frozen
      | otherwise = do
        let (batch, stdGen') = sample stdGen batchSize dataset

        -- Freeze mutable vector of parameters. The frozen version is
        -- then supplied to external mkGrad function provided by user.
        frozen <- U.unsafeFreeze x
        notify frozen k

        -- let grad = M.unionsWith (<+>) (map (mkGrad frozen) batch)
        let grad = parUnions (map (mkGrad frozen) batch)
        addUp grad u
        scale (gain k) u

        x' <- U.unsafeThaw frozen
        apply u x'
        doIt u (k+1) stdGen' x'

-- | Add up all gradients and store results in normal domain.
{-# SPECIALIZE addUp :: Grad -> MVect IO -> IO () #-}
{-# SPECIALIZE addUp :: Grad -> MVect (ST s) -> ST s () #-}
addUp :: Prim.PrimMonad m => Grad -> MVect m -> m ()
addUp grad v = do
    UM.set v 0
    forM_ (toList grad) $ \(i, x) -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (x + y)

-- | Scale the vector by the given value.
{-# SPECIALIZE scale :: Double -> MVect IO -> IO () #-}
{-# SPECIALIZE scale :: Double -> MVect (ST s) -> ST s () #-}
scale :: Prim.PrimMonad m => Double -> MVect m -> m ()
scale c v = do
    forM_ [0 .. UM.length v - 1] $ \i -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (c * y)

-- | Apply gradient to the parameters vector, that is add the first vector to
-- the second one.
{-# SPECIALIZE apply :: MVect IO -> MVect IO -> IO () #-}
{-# SPECIALIZE apply :: MVect (ST s) -> MVect (ST s) -> ST s () #-}
apply :: Prim.PrimMonad m => MVect m -> MVect m -> m ()
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
