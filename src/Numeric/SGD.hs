{-# LANGUAGE RecordWildCards #-}


-- | Stochastic gradient descent implementation using mutable
-- vectors for efficient update of the parameters vector.
-- A user is provided with the immutable vector of parameters
-- so he is able to compute the gradient outside of the IO monad.
-- Currently only the Gaussian priors are implemented.
--
-- This is a preliminary version of the SGD library and API may change
-- in future versions.


module Numeric.SGD
( SgdArgs (..)
, sgdArgsDefault
, Para
, sgd
, module Numeric.SGD.Grad
, module Numeric.SGD.Dataset
) where


import           Control.Monad (forM_)
import qualified System.Random as R
import qualified Data.Vector.Unboxed as U
import qualified Data.Vector.Unboxed.Mutable as UM
import qualified Control.Monad.Primitive as Prim

import           Numeric.SGD.Grad
import           Numeric.SGD.Dataset


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


-- | Vector of parameters.
type Para       = U.Vector Double


-- | Type synonym for mutable vector with Double values.
type MVect      = UM.MVector (Prim.PrimState IO) Double


-- | A stochastic gradient descent method.
-- A notification function can be used to provide user with
-- information about the progress of the learning.
sgd
    :: SgdArgs                  -- ^ SGD parameter values
    -> (Para -> Int -> IO ())   -- ^ Notification run every update
    -> (Para -> x -> Grad)      -- ^ Gradient for dataset element
    -> Dataset x                -- ^ Dataset
    -> Para                     -- ^ Starting point
    -> IO Para                  -- ^ SGD result
sgd SgdArgs{..} notify mkGrad dataset x0 = do
  u <- UM.new (U.length x0)
  doIt u 0 (R.mkStdGen 0) =<< U.thaw x0
  where
    -- Gain in k-th iteration.
    gain k = (gain0 * tau) / (tau + done k)

    -- Number of completed iterations over the full dataset.
    done :: Int -> Double
    done k
        = fromIntegral (k * batchSize)
        / fromIntegral (size dataset)
    -- doneTotal :: Int -> Int
    -- doneTotal = floor . done

    -- Regularization (Guassian prior)
    regularization k = regCoef
      where
        regCoef = (1.0 - gain k * iVar) ** coef
        iVar = 1.0 / regVar
        coef = fromIntegral batchSize
             / fromIntegral (size dataset)

--     -- Regularization (Guassian prior) after a full dataset pass
--     regularization k = 1.0 - (gain k / regVar)

    doIt u k stdGen x
      | done k > iterNum = do
        frozen <- U.unsafeFreeze x
        notify frozen k
        return frozen
      | otherwise = do
        (batch, stdGen') <- sample stdGen batchSize dataset

        -- Regularization
        -- when (doneTotal (k - 1) /= doneTotal k) $ do
        --   <- we now apply regularization each step rather than each
        --      dataset pass
        let regParam = regularization k
        -- putStrLn $ "\nApplying regularization (params *= " ++ show regParam ++ ")"
        scale regParam x

--         -- Regularization
--         when (doneTotal (k - 1) /= doneTotal k) $ do
--           let regParam = regularization k
--           putStrLn $ "\nApplying regularization (params *= " ++ show regParam ++ ")"
--           scale regParam x

        -- Freeze mutable vector of parameters. The frozen version is
        -- then supplied to external mkGrad function provided by user.
        frozen <- U.unsafeFreeze x
        notify frozen k

        -- let grad = M.unions (map (mkGrad frozen) batch)
        let grad = parUnions (map (mkGrad frozen) batch)
        addUp grad u
        scale (gain k) u

        x' <- U.unsafeThaw frozen
        u `addTo` x'
        doIt u (k+1) stdGen' x'


-- | Add up all gradients and store results in normal domain.
addUp :: Grad -> MVect -> IO ()
addUp grad v = do
    UM.set v 0
    forM_ (toList grad) $ \(i, x) -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (x + y)


-- | Scale the vector by the given value.
scale :: Double -> MVect -> IO ()
scale c v = do
    forM_ [0 .. UM.length v - 1] $ \i -> do
        y <- UM.unsafeRead v i
        UM.unsafeWrite v i (c * y)


-- | Apply gradient to the parameters vector, that is add the first vector to
-- the second one.
addTo :: MVect -> MVect -> IO ()
addTo w v = do
    forM_ [0 .. UM.length v - 1] $ \i -> do
        x <- UM.unsafeRead v i
        y <- UM.unsafeRead w i
        UM.unsafeWrite v i (x + y)
