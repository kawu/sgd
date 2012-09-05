{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- | Module provides data type for signed log-domain calculations.

module Numeric.SGD.LogSigned
( LogSigned (..)
, logSigned
, fromPos
, fromNeg
, toNorm
) where

import qualified Data.Number.LogFloat as L
import Control.Monad.Par (NFData)

instance NFData L.LogFloat

-- | Signed real value in the logarithmic domain.
data LogSigned = LogSigned
    { pos :: {-# UNPACK #-} !L.LogFloat     -- ^ Positive component
    , neg :: {-# UNPACK #-} !L.LogFloat     -- ^ Negative component
    }

-- All fields are strict and unpacked.
instance NFData LogSigned

-- | Smart LogSigned constructor.
{-# INLINE logSigned #-}
logSigned :: Double -> LogSigned
logSigned x
    | x > 0     = LogSigned (L.logFloat x) zero
    | x < 0     = LogSigned zero (L.logFloat (-x))
    | otherwise = LogSigned zero zero

-- | Make LogSigned from a positive, log-domain number.
{-# INLINE fromPos #-}
fromPos :: L.LogFloat -> LogSigned
fromPos x = LogSigned x zero

-- | Make LogSigned from a negative, log-domain number.
{-# INLINE fromNeg #-}
fromNeg :: L.LogFloat -> LogSigned
fromNeg x = LogSigned zero x

-- | Shift LogSigned to a normal domain.
{-# INLINE toNorm #-}
toNorm :: LogSigned -> Double
toNorm (LogSigned x y) = L.fromLogFloat x - L.fromLogFloat y

instance Num LogSigned where
    LogSigned x y + LogSigned x' y' =
        LogSigned (x + x') (y + y')
    LogSigned x y * LogSigned x' y' =
        LogSigned (x*x' + y*y') (x*y' + y*x')
    LogSigned x y - LogSigned x' y' =
        LogSigned (x + y') (y + x')
    negate  (LogSigned x y) = LogSigned y x
    abs     (LogSigned x y)
        | x >= y    = LogSigned x y
        | otherwise = LogSigned y x
    signum (LogSigned x y)
        | x > y     =  1
        | x < y     = -1
        | otherwise =  0
    fromInteger = logSigned . fromInteger

{-# INLINE zero #-}
zero :: L.LogFloat
zero = L.logFloat (0 :: Double)
