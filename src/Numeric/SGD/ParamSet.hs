{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE EmptyCase #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveGeneric #-}


module Numeric.SGD.ParamSet
  ( ParamSet(..)
  ) where


import           GHC.Generics
import           GHC.TypeNats (KnownNat)

import           Prelude hiding (div)

import qualified Data.Map.Strict as M

import qualified Numeric.LinearAlgebra.Static as LA


-- | Class of types that can be treated as parameter sets.  It provides basic
-- element-wise operations (addition, multiplication, mapping) which are
-- required to perform stochastic gradient descent.  Many of the operations
-- (`add`, `mul`, `sub`, `div`, etc.) have the same interpretation and follow
-- the same laws (e.g. associativity) as the corresponding operations in `Num`
-- and `Fractional`.  
-- 
-- `zero` takes a parameter set as argument and "zero out"'s all its elements
-- (as in the backprop library).  This allows instances for `Maybe`, `M.Map`,
-- etc., where the structure of the parameter set is dynamic.  This leads to
-- the following property:
--
--     @add (zero x) x = x@
--
-- However, `zero` does not have to obey @(add (zero x) y = y)@.
--
-- A `ParamSet` can be also seen as a (structured) vector, hence `pmap` and
-- `norm_2`.  The latter is not strictly necessary to perform SGD, but it is
-- useful to control the training process.
--
-- `pmap` should obey the following law:
--
--     @pmap id x = x@
--
-- If you leave the body of an instance declaration blank, GHC Generics will be
-- used to derive instances if the type has a single constructor and each field
-- is an instance of `ParamSet`.
class ParamSet a where
  -- | Element-wise mapping
  pmap :: (Double -> Double) -> a -> a

  -- | Zero-out all elements
  zero :: a -> a
  zero = pmap (const 0.0)

--   -- | Element-wise negation
--   neg :: a -> a
--   neg = pmap (\x -> -x)

  -- | Element-wise addition
  add :: a -> a -> a
  -- | Elementi-wise substruction
  sub :: a -> a -> a

  -- | Element-wise multiplication
  mul :: a -> a -> a
  -- | Element-wise division
  div :: a -> a -> a

  -- | L2 norm
  norm_2 :: a -> Double

--   default zero :: (Generic a, GZero (Rep a)) => a -> a
--   zero = genericZero
--   {-# INLINE zero #-}

  default pmap
    :: (Generic a, GPMap (Rep a))
    => (Double -> Double) -> a -> a
  pmap = genericPMap
  {-# INLINE pmap #-}

  default add :: (Generic a, GAdd (Rep a)) => a -> a -> a
  add = genericAdd
  {-# INLINE add #-}

  default sub :: (Generic a, GSub (Rep a)) => a -> a -> a
  sub = genericSub
  {-# INLINE sub #-}

  default mul :: (Generic a, GMul (Rep a)) => a -> a -> a
  mul = genericMul
  {-# INLINE mul #-}

  default div :: (Generic a, GDiv (Rep a)) => a -> a -> a
  div = genericDiv
  {-# INLINE div #-}

  default norm_2 :: (Generic a, GNorm2 (Rep a)) => a -> Double
  norm_2 = genericNorm2
  {-# INLINE norm_2 #-}


-- -- | 'add' using GHC Generics; works if all fields are instances of
-- -- 'ParamSet', but only for values with single constructors.
-- genericZero :: (Generic a, GZero (Rep a)) => a -> a
-- genericZero x = to $ gzero (from x)
-- {-# INLINE genericZero #-}


-- | 'add' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericAdd :: (Generic a, GAdd (Rep a)) => a -> a -> a
genericAdd x y = to $ gadd (from x) (from y)
{-# INLINE genericAdd #-}


-- | 'sub' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericSub :: (Generic a, GSub (Rep a)) => a -> a -> a
genericSub x y = to $ gsub (from x) (from y)
{-# INLINE genericSub #-}


-- | 'div' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericDiv :: (Generic a, GDiv (Rep a)) => a -> a -> a
genericDiv x y = to $ gdiv (from x) (from y)
{-# INLINE genericDiv #-}


-- | 'mul' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericMul :: (Generic a, GMul (Rep a)) => a -> a -> a
genericMul x y = to $ gmul (from x) (from y)
{-# INLINE genericMul #-}


-- | 'norm_2' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericNorm2 :: (Generic a, GNorm2 (Rep a)) => a -> Double
genericNorm2 x = gnorm_2 (from x)
{-# INLINE genericNorm2 #-}


-- | 'pmap' using GHC Generics; works if all fields are instances of
-- 'ParamSet', but only for values with single constructors.
genericPMap :: (Generic a, GPMap (Rep a)) => (Double -> Double) -> a -> a
genericPMap f x = to $ gpmap f (from x)
{-# INLINE genericPMap #-}


--------------------------------------------------
-- Generics
--
-- Partially borrowed from the backprop library
--------------------------------------------------


-- -- | Helper class for automatically deriving 'add' using GHC Generics.
-- class GZero f where
--     gzero :: f t -> f t
-- 
-- instance ParamSet p => GZero (K1 i p) where
--     gzero (K1 x) = K1 (zero x)
--     {-# INLINE gzero #-}
-- 
-- instance (GZero f, GZero g) => GZero (f :*: g) where
--     gzero (x1 :*: y1) = x2 :*: y2
--       where
--         !x2 = gzero x1
--         !y2 = gzero y1
--     {-# INLINE gzero #-}
-- 
-- instance GZero V1 where
--     gzero = \case {}
--     {-# INLINE gzero #-}
-- 
-- instance GZero U1 where
--     gzero _ = U1
--     {-# INLINE gzero #-}
-- 
-- instance GZero f => GZero (M1 i c f) where
--     gzero (M1 x) = M1 (gzero x)
--     {-# INLINE gzero #-}
-- 
-- -- instance GZero f => GZero (f :.: g) where
-- --     gzero = Comp1 gzero
-- --     {-# INLINE gzero #-}


-- | Helper class for automatically deriving 'add' using GHC Generics.
class GAdd f where
    gadd :: f t -> f t -> f t

instance ParamSet a => GAdd (K1 i a) where
    gadd (K1 x) (K1 y) = K1 (add x y)
    {-# INLINE gadd #-}

instance (GAdd f, GAdd g) => GAdd (f :*: g) where
    gadd (x1 :*: y1) (x2 :*: y2) = x3 :*: y3
      where
        !x3 = gadd x1 x2
        !y3 = gadd y1 y2
    {-# INLINE gadd #-}

instance GAdd V1 where
    gadd = \case {}
    {-# INLINE gadd #-}

instance GAdd U1 where
    gadd _ _ = U1
    {-# INLINE gadd #-}

instance GAdd f => GAdd (M1 i c f) where
    gadd (M1 x) (M1 y) = M1 (gadd x y)
    {-# INLINE gadd #-}

-- instance GAdd f => GAdd (f :.: g) where
--     gadd (Comp1 x) (Comp1 y) = Comp1 (gadd x y)
--     {-# INLINE gadd #-}


-- | Helper class for automatically deriving 'sub' using GHC Generics.
class GSub f where
    gsub :: f t -> f t -> f t

instance ParamSet a => GSub (K1 i a) where
    gsub (K1 x) (K1 y) = K1 (sub x y)
    {-# INLINE gsub #-}

instance (GSub f, GSub g) => GSub (f :*: g) where
    gsub (x1 :*: y1) (x2 :*: y2) = x3 :*: y3
      where
        !x3 = gsub x1 x2
        !y3 = gsub y1 y2
    {-# INLINE gsub #-}

instance GSub V1 where
    gsub = \case {}
    {-# INLINE gsub #-}

instance GSub U1 where
    gsub _ _ = U1
    {-# INLINE gsub #-}

instance GSub f => GSub (M1 i c f) where
    gsub (M1 x) (M1 y) = M1 (gsub x y)
    {-# INLINE gsub #-}

-- instance GSub f => GSub (f :.: g) where
--     gsub (Comp1 x) (Comp1 y) = Comp1 (gsub x y)
--     {-# INLINE gsub #-}


-- | Helper class for automatically deriving 'mul' using GHC Generics.
class GMul f where
    gmul :: f t -> f t -> f t

instance ParamSet a => GMul (K1 i a) where
    gmul (K1 x) (K1 y) = K1 (mul x y)
    {-# INLINE gmul #-}

instance (GMul f, GMul g) => GMul (f :*: g) where
    gmul (x1 :*: y1) (x2 :*: y2) = x3 :*: y3
      where
        !x3 = gmul x1 x2
        !y3 = gmul y1 y2
    {-# INLINE gmul #-}

instance GMul V1 where
    gmul = \case {}
    {-# INLINE gmul #-}

instance GMul U1 where
    gmul _ _ = U1
    {-# INLINE gmul #-}

instance GMul f => GMul (M1 i c f) where
    gmul (M1 x) (M1 y) = M1 (gmul x y)
    {-# INLINE gmul #-}

-- instance GMul f => GMul (f :.: g) where
--     gmul (Comp1 x) (Comp1 y) = Comp1 (gmul x y)
--     {-# INLINE gmul #-}


-- | Helper class for automatically deriving 'div' using GHC Generics.
class GDiv f where
    gdiv :: f t -> f t -> f t

instance ParamSet a => GDiv (K1 i a) where
    gdiv (K1 x) (K1 y) = K1 (div x y)
    {-# INLINE gdiv #-}

instance (GDiv f, GDiv g) => GDiv (f :*: g) where
    gdiv (x1 :*: y1) (x2 :*: y2) = x3 :*: y3
      where
        !x3 = gdiv x1 x2
        !y3 = gdiv y1 y2
    {-# INLINE gdiv #-}

instance GDiv V1 where
    gdiv = \case {}
    {-# INLINE gdiv #-}

instance GDiv U1 where
    gdiv _ _ = U1
    {-# INLINE gdiv #-}

instance GDiv f => GDiv (M1 i c f) where
    gdiv (M1 x) (M1 y) = M1 (gdiv x y)
    {-# INLINE gdiv #-}

-- instance GDiv f => GDiv (f :.: g) where
--     gdiv (Comp1 x) (Comp1 y) = Comp1 (gdiv x y)
--     {-# INLINE gdiv #-}


-- | Helper class for automatically deriving 'norm_2' using GHC Generics.
class GNorm2 f where
    gnorm_2 :: f t -> Double

instance ParamSet a => GNorm2 (K1 i a) where
    gnorm_2 (K1 x) = norm_2 x
    {-# INLINE gnorm_2 #-}

instance (GNorm2 f, GNorm2 g) => GNorm2 (f :*: g) where
    gnorm_2 (x1 :*: y1) =
      sqrt ((x2 ^ (2 :: Int)) + (y2 ^ (2 :: Int)))
      where
        !x2 = gnorm_2 x1
        !y2 = gnorm_2 y1
    {-# INLINE gnorm_2 #-}

instance GNorm2 V1 where
    gnorm_2 = \case {}
    {-# INLINE gnorm_2 #-}

instance GNorm2 U1 where
    gnorm_2 _ = 0
    {-# INLINE gnorm_2 #-}

instance GNorm2 f => GNorm2 (M1 i c f) where
    gnorm_2 (M1 x) = gnorm_2 x
    {-# INLINE gnorm_2 #-}

-- -- TODO: Make sure this makes sense
-- instance GNorm2 f => GNorm2 (f :.: g) where
--     gnorm_2 (Comp1 x) = gnorm_2 x
--     {-# INLINE gnorm_2 #-}


-- | Helper class for automatically deriving 'pmap' using GHC Generics.
class GPMap f where
    gpmap :: (Double -> Double) -> f t -> f t

instance ParamSet a => GPMap (K1 i a) where
    gpmap f (K1 x) = K1 (pmap f x)
    {-# INLINE gpmap #-}

instance (GPMap f, GPMap g) => GPMap (f :*: g) where
    gpmap f (x1 :*: y1) = x2 :*: y2
      where
        !x2 = gpmap f x1
        !y2 = gpmap f y1
    {-# INLINE gpmap #-}

instance GPMap V1 where
    gpmap _ = \case {}
    {-# INLINE gpmap #-}

instance GPMap U1 where
    gpmap _ _ = U1
    {-# INLINE gpmap #-}

instance GPMap f => GPMap (M1 i c f) where
    gpmap f (M1 x) = M1 (gpmap f x)
    {-# INLINE gpmap #-}

-- instance GPMap f => GPMap (f :.: g) where
--     gpmap f (Comp1 x) = Comp1 (gpmap f x)
--     {-# INLINE gpmap #-}


--------------------------------------------------
-- Basic instances
--------------------------------------------------


instance ParamSet Double where
  zero = const 0
  pmap = id
  add = (+)
  sub = (-)
  mul = (*)
  div = (/)
  norm_2 = id


instance (KnownNat n) => ParamSet (LA.R n) where
  zero = const 0
  pmap = LA.dvmap
  add = (+)
  sub = (-)
  mul = (*)
  div = (/)
  norm_2 = LA.norm_2


instance (KnownNat n, KnownNat m) => ParamSet (LA.L n m) where
  zero = const 0
  pmap = LA.dmmap
  add = (+)
  sub = (-)
  mul = (*)
  div = (/)
  norm_2 = LA.norm_2


-- | `Nothing` represents a deactivated parameter set component. If `Nothing`
-- is given as an argument to one of the `ParamSet` operations, the result is
-- `Nothing` as well.
--
-- This differs from the corresponding instance in the backprop library, where
-- `Nothing` is equivalent to `Just 0`.  However, the implementation below
-- seems to correspond adequately enough to the notion that a particular
-- component is either active or not in both the parameter set and the
-- gradient, hence it doesn't make sense to combine `Just` with `Nothing`.
instance (ParamSet a) => ParamSet (Maybe a) where
  zero = fmap zero
  pmap = fmap . pmap

  add (Just x) (Just y) = Just (add x y)
  add _ _ = Nothing

  sub (Just x) (Just y) = Just (sub x y)
  sub _ _ = Nothing

  mul (Just x) (Just y) = Just (mul x y)
  mul _ _ = Nothing

  div (Just x) (Just y) = Just (div x y)
  div _ _ = Nothing

  norm_2 = maybe 0 norm_2


-- | A map with different parameter sets (of the same type) assigned to the
-- individual keys.
--
-- When combining two maps with different sets of keys, only their intersection
-- is preserved.
instance (Ord k, ParamSet a) => ParamSet (M.Map k a) where
  zero = fmap zero
  pmap f = fmap (pmap f)
  add = M.intersectionWith add
  sub = M.intersectionWith sub
  mul= M.intersectionWith mul
  div= M.intersectionWith div
  norm_2 = sqrt . sum . map ((^(2::Int)) . norm_2)  . M.elems
