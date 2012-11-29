{-# LANGUAGE ForeignFunctionInterface #-}

module Numeric.SGD.Grad.Vector
( Grad
, partition
) where

import Control.Applicative ((<$>))
import Control.Monad (forM_, unless)
import Control.Concurrent.Async (mapConcurrently)
import Control.Monad.ST (RealWorld)
import System.IO.Unsafe (unsafeInterleaveIO)
import GHC.Conc (numCapabilities)
import qualified Data.Vector.Unboxed.Mutable as V

import qualified Numeric.SGD.LogSigned as L
import qualified Data.Number.LogFloat as L
import qualified Numeric.SGD.Grad as G

foreign import ccall unsafe "math.h log1p"
    log1p :: Double -> Double

type LogSigned = (Double, Double)

zero :: LogSigned
zero = (-(1/0), -(1/0))
{-# INLINE zero #-}

plainZero :: LogSigned -> Bool
plainZero = (==zero)
{-# INLINE plainZero #-}

fromExt :: L.LogSigned -> LogSigned
fromExt x =
    ( L.logFromLogFloat (L.pos x)
    , L.logFromLogFloat (L.neg x) )
{-# INLINE fromExt #-}

toExt :: LogSigned -> L.LogSigned
toExt (x, y) = L.LogSigned
    (L.logToLogFloat x)
    (L.logToLogFloat y)
{-# INLINE toExt #-}

addLog :: Double -> Double -> Double
addLog x y
    | x == y
      && isInfinite x
      && isInfinite y = x -- @0+0 == 0@, @infinity+infinity == infinity@
    | x >= y          = x + log1p (exp (y - x))
    | otherwise       = y + log1p (exp (x - y))
{-# INLINE addLog #-}

addSigned :: LogSigned -> LogSigned -> LogSigned
addSigned (x, y) (x', y') = (addLog x x', addLog y y')
{-# INLINE addSigned #-}

newtype Grad = Grad { unGrad :: V.MVector RealWorld LogSigned }

instance G.GradIO Grad where
    empty k             = do
        x <- Grad <$> V.new k
        G.clear x
        return x
    clear (Grad v)      = V.set v zero
    fill xs (Grad v)    = do
        forM_ xs $ \(i, y) -> do
            x <- V.read v i
            V.write v i (addSigned x (fromExt y))
        return (Grad v)
    content (Grad v)    = do
        doIt 0
      where
        doIt i
            | i < n = do
                x <- V.unsafeRead v i
                if plainZero x
                    then doIt (i + 1)
                    else do
                        xs <- unsafeInterleaveIO $ doIt (i + 1)
                        return $ (i, toExt x) : xs
            | otherwise = return []
        n = V.length v
    unionsTo grads (Grad v) = do
        _ <- mapConcurrently
            (uncurry doIt)
            (partition numCapabilities 0 (V.length v))
        return (Grad v)
      where
        vs = map unGrad grads
        doIt i j = forM_ vs $ \w -> do
            forM_ [i..j-1] $ \k -> do
                y <- V.unsafeRead w k
                unless (plainZero y) $ do
                    x <- V.unsafeRead v k
                    V.unsafeWrite v k (addSigned x y)

partition :: Int -> Int -> Int -> [(Int, Int)]
partition k p q =
    [(i, min (i + step) q) | i <- ps]
  where
    step = ((q - p) `div` k) + 1
    ps   = takeWhile (<q) [p, p + step ..]
