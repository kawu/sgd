{-# LANGUAGE CPP #-}

-- | A gradient is represented by an IntMap from gradient indices to values.
-- Elements with no associated values in the gradient are assumed to have a 0
-- value assigned.  Such elements are of no interest: when adding the gradient
-- to the vector of parameters, only non-zero elements are taken into account.
-- 
-- Each value associated with a gradient position is a pair of positive and
-- negative components.  They are stored separately to ensure high accuracy of
-- computation results.  Besides, both positive and negative components are
-- stored in a logarithmic domain.

module Numeric.SGD.Sparse.Grad
( Grad
, empty
, add
, addL
, fromList
, fromLogList
, toList
, parUnions
) where

import Data.List (foldl')
import Control.Applicative ((<$>), (<*>))
import Control.Monad.Par (Par, runPar, get)
#if MIN_VERSION_containers(0,4,2)
import Control.Monad.Par (spawn)
#else
import Control.DeepSeq (deepseq)
import Control.Monad.Par (spawn_)
#endif
#if MIN_VERSION_containers(0,5,0)
import qualified Data.IntMap.Strict as M
#else
import qualified Data.IntMap as M
#endif

import Numeric.SGD.Sparse.LogSigned

-- | Gradient with nonzero values stored in a logarithmic domain.
-- Since values equal to zero have no impact on the update phase
-- of the SGD method, it is more efficient to not to store those
-- components in the gradient.
type Grad = M.IntMap LogSigned

{-# INLINE insertWith #-}
insertWith :: (a -> a -> a) -> M.Key -> a -> M.IntMap a -> M.IntMap a
#if MIN_VERSION_containers(0,5,0)
insertWith = M.insertWith
#elif MIN_VERSION_containers(0,4,1)
insertWith = M.insertWith'
#else
insertWith f k x m = 
    M.alter g k m
  where
    g my = case my of
        Nothing -> Just x
        Just y  ->
            let z = f x y
            in  z `seq` Just z
#endif

-- | Add normal-domain double to the gradient at the given position.
{-# INLINE add #-}
add :: Grad -> Int -> Double -> Grad
add grad i y = insertWith (+) i (logSigned y) grad 


-- | Add log-domain, singed number to the gradient at the given position.
{-# INLINE addL #-}
addL :: Grad -> Int -> LogSigned -> Grad
addL grad i y = insertWith (+) i y grad 

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

-- | Parallel unions in the Par monad.
parUnionsP :: [Grad] -> Par Grad
parUnionsP [x] = return x
parUnionsP zs  = do
    let (xs, ys) = split zs
#if MIN_VERSION_containers(0,4,2)
    xsP <- spawn (parUnionsP xs)
    ysP <- spawn (parUnionsP ys)
    M.unionWith (+) <$> get xsP <*> get ysP
#else
    xsP <- spawn_ (parUnionsP xs)
    ysP <- spawn_ (parUnionsP ys)
    x <- M.unionWith (+) <$> get xsP <*> get ysP
    M.elems x `deepseq` return x
#endif
  where
    split []        = ([], [])
    split (x:[])    = ([x], [])
    split (x:y:rest)  =
        let (xs, ys) = split rest
        in  (x:xs, y:ys)
