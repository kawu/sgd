{-# LANGUAGE RecordWildCards #-}


-- | A version of `Numeric.SGD` extended with momentum.


module Numeric.SGD.Momentum
( SgdArgs (..)
, sgdArgsDefault
, Para
, sgd
, module Numeric.SGD.Grad
, module Numeric.SGD.Dataset
) where


import           Control.Monad (forM_, when)
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

  putStrLn $ "Running momentum!"

  -- A vector for the momentum gradient
  momentum <- UM.new (U.length x0)

  -- A worker vector for computing the actual gradients
  u <- UM.new (U.length x0)

  doIt momentum u 0 (R.mkStdGen 0) =<< U.thaw x0

  where
    -- Gain in k-th iteration.
    gain k = (gain0 * tau) / (tau + done k)

    -- Number of completed iterations over the full dataset.
    done :: Int -> Double
    done k
        = fromIntegral (k * batchSize)
        / fromIntegral (size dataset)
    doneTotal :: Int -> Int
    doneTotal = floor . done

    -- Regularization (Guassian prior) parameter
    regularizationParam = regCoef
      where
        regCoef = iVar ** coef
        iVar = 1.0 / regVar
        coef = fromIntegral (size dataset)
             / fromIntegral batchSize

    -- The gamma parameter. TODO: put in SgdArgs.
    gamma = 0.9

    doIt momentum u k stdGen x

      | done k > iterNum = do
        frozen <- U.unsafeFreeze x
        notify frozen k
        return frozen

      | otherwise = do

        -- Sample the dataset
        (batch, stdGen') <- sample stdGen batchSize dataset

        -- NEW: comment out
        -- -- Apply regularization to the parameters vector.
        -- scale (regularization k) x

        -- Freeze mutable vector of parameters. The frozen version is
        -- then supplied to external mkGrad function provided by user.
        frozen <- U.unsafeFreeze x
        notify frozen k

        -- Compute the gradient and put it in `u`
        let grad = parUnions (map (mkGrad frozen) batch)
        addUp grad u

        -- Apply regularization to `u`
        applyRegularization regularizationParam x u

        -- Scale the gradient
        scale (gain k) u

        -- Compute the new momentum
        updateMomentum gamma momentum u

        x' <- U.unsafeThaw frozen
        momentum `addTo` x'
        doIt momentum u (k+1) stdGen' x'


-- | Compute the new momentum (gradient) vector.
applyRegularization
  :: Double -- ^ Regularization parameter
  -> MVect  -- ^ The parameters
  -> MVect  -- ^ The current gradient
  -> IO ()
applyRegularization regParam params grad = do
  forM_ [0 .. UM.length grad - 1] $ \i -> do
    x <- UM.unsafeRead grad i
    y <- UM.unsafeRead params i
    UM.unsafeWrite grad i $ x - regParam * y


-- | Compute the new momentum (gradient) vector.
updateMomentum
  :: Double -- ^ The gamma parameter
  -> MVect  -- ^ The previous momentum
  -> MVect  -- ^ The scaled current gradient
  -> IO ()
updateMomentum gamma momentum grad = do
  forM_ [0 .. UM.length momentum - 1] $ \i -> do
    x <- UM.unsafeRead momentum i
    y <- UM.unsafeRead grad i
    UM.unsafeWrite momentum i (gamma * x + y)


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
