{-# LANGUAGE GeneralizedNewtypeDeriving #-}

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

module Numeric.SGD.Grad.Map
( Grad
) where

import Data.List (foldl')
import Control.Applicative ((<$>), (<*>))
import Control.Monad.Par.Scheds.Direct (Par, runPar, get)
import Control.Monad.Par.Scheds.Direct (spawn)
import Control.DeepSeq (NFData)
import qualified Data.IntMap as M

import Numeric.SGD.LogSigned
import qualified Numeric.SGD.Grad as G

-- | Gradient with nonzero values stored in a logarithmic domain.
-- Since values equal to zero have no impact on the update phase
-- of the SGD method, it is more efficient to not to store those
-- components in the gradient.
newtype Grad = Grad { unGrad :: M.IntMap LogSigned }
    deriving (Show, Eq, Ord, NFData)

-- | Add log-domain, singed number to the gradient at the given position.
addL :: Grad -> Int -> LogSigned -> Grad
addL (Grad v) i y = Grad $ M.insertWith' (+) i y v
{-# INLINE addL #-}

instance G.Grad Grad where
    fromList =
        let ins grad (i, y) = addL grad i y
        in  foldl' ins (Grad M.empty)

    toList = M.assocs . unGrad

    unions [] = error "parUnions: empty list"
    unions xs = runPar (parUnionsP xs)

instance G.GradIO Grad where
    empty _         = return (G.fromList [])
    clear _         = return ()
    fill xs _       = return (G.fromList xs)
    content v       = return (G.toList v)
    unionsTo vs v   = return (G.unions (v:vs))

union :: Grad -> Grad -> Grad
union (Grad v) (Grad w) = Grad $ M.unionWith (+) v w

-- | Parallel unions in the Par monad.
parUnionsP :: [Grad] -> Par Grad
parUnionsP [x] = return x
parUnionsP zs  = do
    let (xs, ys) = split zs
    xsP <- spawn (parUnionsP xs)
    ysP <- spawn (parUnionsP ys)
    union <$> get xsP <*> get ysP
  where
    split []        = ([], [])
    split (x:[])    = ([x], [])
    split (x:y:rest)  =
        let (xs, ys) = split rest
        in  (x:xs, y:ys)
