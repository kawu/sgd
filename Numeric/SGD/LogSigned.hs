{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

-- | Module provides data type for signed log-domain calculations.

module Numeric.SGD.LogSigned
( LogSigned (..)
, logSigned
, fromPos
, fromNeg
, toNorm
, toLogFloat
) where

import qualified Data.Number.LogFloat as L
import Data.Function (on)
import Control.DeepSeq (NFData(..))

-- | Signed real value in the logarithmic domain.
data LogSigned = LogSigned
    { pos :: {-# UNPACK #-} !L.LogFloat     -- ^ Positive component
    , neg :: {-# UNPACK #-} !L.LogFloat     -- ^ Negative component
    } deriving Show

instance Eq LogSigned where
    (==) = (==) `on` toLogFloat

instance Ord LogSigned where
    compare = compare `on` toLogFloat

-- All fields are strict and unpacked.
instance NFData LogSigned where
    rnf (LogSigned p q) = p `seq` q `seq` ()

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

-- | Change the 'LogSigned' to either negative 'Left' 'L.LogFloat'
-- or positive 'Right' 'L.LogFloat'.
toLogFloat :: LogSigned -> Either L.LogFloat L.LogFloat
toLogFloat x = case signum x of
    -1  -> Left  $ neg x - pos x
    1   -> Right $ pos x - neg x
    _   -> Right $ L.logFloat (0 :: Double)

instance Num LogSigned where
    LogSigned x y + LogSigned x' y' =
        LogSigned (x + x') (y + y')
    {-# INLINE (+) #-}
    LogSigned x y * LogSigned x' y' =
        LogSigned (x*x' + y*y') (x*y' + y*x')
    {-# INLINE (*) #-}
    LogSigned x y - LogSigned x' y' =
        LogSigned (x + y') (y + x')
    {-# INLINE (-) #-}
    negate  (LogSigned x y) = LogSigned y x
    {-# INLINE negate #-}
    abs     (LogSigned x y)
        | x >= y    = LogSigned x y
        | otherwise = LogSigned y x
    {-# INLINE abs #-}
    signum (LogSigned x y)
        | x > y     =  1
        | x < y     = -1
        | otherwise =  0
    {-# INLINE signum #-}
    fromInteger = logSigned . fromInteger
    {-# INLINE fromInteger #-}

zero :: L.LogFloat
zero = L.logFloat (0 :: Double)
{-# INLINE zero #-}
