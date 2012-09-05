-- | A gradient is represented by an IntMap from gradient indices
-- to values. Elements with no associated values in the gradient
-- are assumed to have a 0 value assigned. Such elements are
-- not interesting: when adding the gradient to the vector of
-- parameters, only nonzero elements are taken into account.
-- 
-- Each value associated with a gradient position is a pair of
-- positive and negative components. They are stored separately
-- to ensure high accuracy of computation results.
-- Besides, both positive and negative components are stored
-- in a logarithmic domain.

module Numeric.SGD.Grad
( Grad
, empty
, add
, addL
, fromList
, fromLogList
, toList
, parUnions
) where

import Control.Applicative ((<$>), (<*>))
import Data.List (foldl')
import qualified Data.IntMap as M
import Control.Monad.Par.Scheds.Direct (Par, runPar, spawn, get)

import Numeric.SGD.LogSigned

-- -- | Signed number.
-- data Signed a
--     = Pos a     -- ^ Positive number
--     | Neg a     -- ^ Negative number

-- | Gradient with nonzero values stored in a logarithmic domain.
-- Since values equal to zero have no impact on the update phase
-- of the SGD method, it is more efficient to not to store those
-- components in the gradient.
type Grad = M.IntMap LogSigned

-- | Add normal-domain double to the gradient at the given position.
{-# INLINE add #-}
add :: Grad -> Int -> Double -> Grad
add grad i y = M.insertWith' (+) i (logSigned y) grad 

-- | Add log-domain, singed number to the gradient at the given position.
{-# INLINE addL #-}
addL :: Grad -> Int -> LogSigned -> Grad
addL grad i y = M.insertWith' (+) i y grad 

-- | Construct gradient from a list of (index, value) pairs.
-- All values from the list are added at respective gradient
-- positions.
{-# INLINE fromList #-}
fromList :: [(Int, Double)] -> Grad
fromList =
    let ins grad (i, y) = add grad i y
    in  foldl' ins empty

-- | Construct gradient from a list of (index, signed, log-domain number)
-- pairs.  All values from the list are added at respective gradient
-- positions.
{-# INLINE fromLogList #-}
fromLogList :: [(Int, LogSigned)] -> Grad
fromLogList =
    let ins grad (i, y) = addL grad i y
    in  foldl' ins empty

-- | Collect gradient components with values in normal domain.
{-# INLINE toList #-}
toList :: Grad -> [(Int, Double)]
toList =
    let unLog (i, x) = (i, toNorm x)
    in  map unLog . M.assocs

-- | Empty gradient, i.e. with all elements set to 0.
{-# INLINE empty #-}
empty :: Grad
empty = M.empty

-- | Perform parallel unions operation on gradient list. 
-- Experimental version.
parUnions :: [Grad] -> Grad
parUnions [] = error "parUnions: empty list"
parUnions xs = runPar (parUnionsP xs)

-- | Parallel unoins in the Par monad.
parUnionsP :: [Grad] -> Par Grad
parUnionsP [x] = return x
parUnionsP zs  = do
    let (xs, ys) = split zs
    xsP <- spawn (parUnionsP xs)
    ysP <- spawn (parUnionsP ys)
    M.unionWith (+) <$> get xsP <*> get ysP
  where
    split []        = ([], [])
    split (x:[])    = ([x], [])
    split (x:y:rest)  =
        let (xs, ys) = split rest
        in  (x:xs, y:ys)
