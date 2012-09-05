{-# LANGUAGE RecordWildCards #-}

import Control.Applicative ((<$>), (<*>))
import Control.Monad (forM_, replicateM)
import System.IO (hSetBuffering, stdout, BufferMode (NoBuffering))
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as U
import qualified Numeric.SGD as S
import qualified Data.IntMap as M

-- | Element of a dataset.
type Elem = [(Int, Double)]

-- | Objective function.
f :: [Elem] -> S.Para -> Double
f xs para = sum $ map (flip fe para) xs

-- | Objective function per data element.
fe :: Elem -> S.Para -> Double
fe xs para = sum
    [ (para U.! k - x) ^ 2
    | (k, x) <- xs ]

-- | Negated gradient computation. Gradient argument may be not
-- zeroed, but we just add new values to it.
update :: S.Para -> Elem -> S.Grad
update para xs = S.fromList
    [ (k, 2 * (x - para U.! k))
    | (k, x) <- xs ]

-- | Random Elem.
elemR
    :: Int              -- ^ Maximum number of element items,
    -> (Int, Int)       -- ^ Range for item's first component,
    -> (Double, Double) -- ^ Range for item's second component,
    -> IO Elem          -- ^ Result.
elemR nMax xr yr = do
    n <- R.randomRIO (0, max 0 nMax)
    replicateM n ((,) <$> R.randomRIO xr <*> R.randomRIO yr)

notify :: S.SgdArgs -> V.Vector Elem -> S.Para -> Int -> IO ()
notify S.SgdArgs{..} dataSet para k =
    if floor (done k) /= floor (done (k - 1))
        then do
            let n = floor (done k)
                x = f (V.toList dataSet) para
            putStrLn ("\n" ++ "[" ++ show n ++ "] f = " ++ show x)
        else
            putStr "."
  where
    done k
        = fromIntegral (k * batchSize)
        / fromIntegral (V.length dataSet)

main = do
    let n = 1000000     -- ^ Parameter number
        m = 1000         -- ^ Dataset size
        k = 1000       -- ^ Maximum number of items in data element
        yr = (-10, 10)
    dataSet <- V.fromList <$> replicateM m (elemR k (0, n-1) yr)
    -- | Initial parameters value.
    let para = U.replicate n 0
    -- | SGD computation.
    hSetBuffering stdout NoBuffering
    let sgdArgs = S.sgdArgsDefault
            { S.gain0 = 0.5
            , S.iterNum = 25
            , S.tau = 5 }
    S.sgdM sgdArgs (notify sgdArgs dataSet) update dataSet para
