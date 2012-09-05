{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- | Module provides data type for signed log-domain calculations.

module Numeric.SGD.LogSigned
( LogSigned (..)
, logSigned
, toNorm
) where

import qualified Data.Number.LogFloat as L
import Control.Monad.Par (NFData)

instance NFData L.LogFloat

-- | Signed real value in logarithmic domain.
-- Positive and negative components are stored in separate fields.
newtype LogSigned = LogSigned { unSigned :: (L.LogFloat, L.LogFloat) }
    deriving (NFData)

{-# INLINE zero #-}
zero :: L.LogFloat
zero = L.logFloat (0 :: Double)

instance Num LogSigned where
    LogSigned (!x, !y) + LogSigned (!x', !y') =
        LogSigned (x + x', y + y')
    LogSigned (!x, !y) * LogSigned (!x', !y') =
        LogSigned (x*x' + y*y', x*y' + y*x')
    LogSigned (!x, !y) - LogSigned (!x', !y') =
        LogSigned (x + y', y + x')
    negate  (LogSigned (!x, !y)) = LogSigned (y, x)
    abs     (LogSigned (!x, !y))
        | x >= y    = LogSigned (x, y)
        | otherwise = LogSigned (y, x)
    signum (LogSigned (!x, !y))
        | x > y     =  1
        | x < y     = -1
        | otherwise =  0
    fromInteger = logSigned . fromInteger

-- | Smart LogSigned constructor.
{-# INLINE logSigned #-}
logSigned :: Double -> LogSigned
logSigned x
    | x > 0     = LogSigned (L.logFloat x, zero)
    | x < 0     = LogSigned (zero, L.logFloat (-x))
    | otherwise = LogSigned (zero, zero)

-- | Shift LogSigned to a normal domain.
{-# INLINE toNorm #-}
toNorm :: LogSigned -> Double
toNorm (LogSigned (x, y)) = L.fromLogFloat x - L.fromLogFloat y
