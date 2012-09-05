{-# LANGUAGE BangPatterns #-}

module Numeric.SGD.Grad
( Grad (..)
, Signed (..)
, empty
, add
, addL
, addPosL
, addNegL
, fromList
, fromLogList
, toList
, parUnions
) where

import Control.Applicative ((<$>), (<*>))
import Data.List (foldl')
import qualified Data.Number.LogFloat as L
import qualified Data.IntMap as M

-- Nested scheduler bug: https://github.com/simonmar/monad-par/issues/21
-- import Control.Monad.Par (Par, runPar, spawn, get, NFData)
import Control.Monad.Par (NFData)
import Control.Monad.Par.Scheds.Direct (Par, runPar, spawn, get)

instance NFData L.LogFloat

-- | Gradient with positive and negative components in log domain.
-- It is represented by an IntMap from indices to nonzero values
-- and can be defined by the formula: v i = exp (pos i) - exp (neg i).
type Grad = M.IntMap (L.LogFloat, L.LogFloat)

-- | Add two log-domain numbers.
{-# INLINE (<+>) #-}
(<+>) :: (L.LogFloat, L.LogFloat)
      -> (L.LogFloat, L.LogFloat)
      -> (L.LogFloat, L.LogFloat)
(!x, !y) <+> (!x', !y') = (x + x', y + y')

{-# INLINE zero #-}
zero :: L.LogFloat
zero = L.logFloat (0 :: Double)

-- | Add normal-domain double to the gradient at the given position.
{-# INLINE add #-}
add :: Grad -> Int -> Double -> Grad
add grad i y =
    M.insertWith' (<+>) i y' grad 
  where
    y' = case y >= 0 of
        True    -> (L.logFloat y, zero)
        False   -> (zero, L.logFloat (-y))

data Signed a
    = Pos a     -- ^ Positive number
    | Neg a     -- ^ Negative number

{-# INLINE addL #-}
addL :: Grad -> Int -> Signed L.LogFloat -> Grad
addL grad i (Pos x) = addPosL grad i x
addL grad i (Neg x) = addNegL grad i x

-- | Add log-domain, positive number to the gradient at the given position.
{-# INLINE addPosL #-}
addPosL :: Grad -> Int -> L.LogFloat -> Grad
addPosL grad i y = M.insertWith' (<+>) i (y, zero) grad

-- | Add log-domain, negative number to the gradient at the given position.
{-# INLINE addNegL #-}
addNegL :: Grad -> Int -> L.LogFloat -> Grad
addNegL grad i y = M.insertWith' (<+>) i (zero, y) grad

{-# INLINE fromList #-}
fromList :: [(Int, Double)] -> Grad
fromList =
    let ins grad (i, y) = add grad i y
    in  foldl' ins empty

{-# INLINE fromLogList #-}
fromLogList :: [(Int, Signed L.LogFloat)] -> Grad
fromLogList =
    let ins grad (i, y) = addL grad i y
    in  foldl' ins empty

{-# INLINE toList #-}
toList :: Grad -> [(Int, Double)]
toList =
    let toNorm (i, (pos, neg)) =
            (i, L.fromLogFloat pos - L.fromLogFloat neg)
    in  map toNorm . M.assocs

{-# INLINE empty #-}
empty :: Grad
empty = M.empty

-- | Perform parallel unions operation on gradient list. 
-- Experimental version.
parUnions :: [Grad] -> Grad
parUnions [] = error "parUnions: empty list"
parUnions xs = runPar (parUnionsP xs)

parUnionsP :: [Grad] -> Par Grad
parUnionsP [x] = return x
parUnionsP zs  = do
    let (xs, ys) = split zs
    xsP <- spawn (parUnionsP xs)
    ysP <- spawn (parUnionsP ys)
    M.unionWith (<+>) <$> get xsP <*> get ysP
  where
    split []        = ([], [])
    split (x:[])    = ([x], [])
    split (x:y:rest)  =
        let (xs, ys) = split rest
        in  (x:xs, y:ys)
