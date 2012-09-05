{-# LANGUAGE RecordWildCards #-}

import Control.Applicative ((<$>), (<*>))
import Control.Monad (replicateM)
import System.IO (hSetBuffering, stdout, BufferMode (NoBuffering))
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import qualified Numeric.SGD as S

------------------------------------------------------------------------------
-- Dataset generation
------------------------------------------------------------------------------

-- | Element of a dataset.
type Elem = [(Int, Double)]

-- | Random dataset element.
elemR
    :: Int              -- ^ Maximum number of element items
    -> (Int, Int)       -- ^ Range for item's first component
    -> (Double, Double) -- ^ Range for item's second component
    -> IO Elem          -- ^ Result
elemR nMax xr yr = do
    n <- R.randomRIO (0, max 0 nMax)
    replicateM n ((,) <$> R.randomRIO xr <*> R.randomRIO yr)

-- | Random dataset.
dataSetR
    :: Int              -- ^ Dataset size
    -> Int              -- ^ Number of model parameters
    -> Int              -- ^ Maximum number of items in data element
    -> (Double, Double) -- ^ Range for item's second component
    -> IO (V.Vector Elem)   -- ^ Result
dataSetR m n k yRan =
    V.fromList <$> replicateM m (elemR k (0, n-1) yRan)

------------------------------------------------------------------------------
-- Objective function and gradient
------------------------------------------------------------------------------

-- | An objective function. The SGD method can be used when
-- the objective function is defined in a form of a sum.
goal :: S.Para -> [Elem] -> Double
goal para =
    sum . map perElem
  where
    perElem xs = sum
        [ (para U.! k - x) ^ (2 :: Int)
        | (k, x) <- xs ]

-- | Since the goal function has a form of a sum, it is sufficient to define
-- the gradient over one element only. The gradient with respect to the dataset
-- is a sum of gradients over its individual elements.
grad :: S.Para -> Elem -> S.Grad
grad para xs = S.fromList
    -- [ (k, 2 * (x - para U.! k))
    [ (k, 2 * (para U.! k - x))
    | (k, x) <- xs ]

-- | Negate gradient. We use it to find the minimum of the objective function.
negGrad :: (S.Para -> Elem -> S.Grad)
        -> (S.Para -> Elem -> S.Grad)
negGrad g para x = fmap negate (g para x)

------------------------------------------------------------------------------
-- SGD
------------------------------------------------------------------------------

-- | Notification run by the sgdM function every parameters update.
notify :: S.SgdArgs -> V.Vector Elem -> S.Para -> Int -> IO ()
notify S.SgdArgs{..} dataSet para k =
    if doneTotal k /= doneTotal (k - 1)
        then do
            let n = doneTotal k
                x = goal para (V.toList dataSet)
            putStrLn ("\n" ++ "[" ++ show n ++ "] f = " ++ show x)
        else
            putStr "."
  where
    doneTotal :: Int -> Int
    doneTotal = floor . done
    done :: Int -> Double
    done i
        = fromIntegral (i * batchSize)
        / fromIntegral (V.length dataSet)

-- | Run the monadic version of SGD.
runSgdM
    :: Int              -- ^ Dataset size
    -> Int              -- ^ Number of model parameters
    -> Int              -- ^ Maximum number of items in data element
    -> S.SgdArgs        -- ^ SGD parameters
    -> IO S.Para
runSgdM m n k sgdArgs = do
    dataSet <- dataSetR m n k (-10, 10)
    let para = U.replicate n 0
    hSetBuffering stdout NoBuffering
    S.sgdM sgdArgs (notify sgdArgs dataSet) (negGrad grad) dataSet para

-- | Run the monadic version of SGD with some default parameter values.
main = do
    let sgdArgs = S.sgdArgsDefault { S.iterNum = 50 }
    runSgdM 1000 1000000 10 sgdArgs
