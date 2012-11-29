module Numeric.SGD.Grad
( Grad (..)
, GradIO (..)
) where

import Numeric.SGD.LogSigned

-- | A pure gradient abstraction.
class Grad v where
    -- | Construct gradient from a list of (position, number to add) pairs.
    fromList    :: [(Int, LogSigned)] -> v
    -- | List representation of a gradient.
    toList      :: v -> [(Int, LogSigned)]
    -- | Unions of a list of gradients.
    unions      :: [v] -> v

-- | A monadic version of gradient.
class GradIO v where
    -- | Empty gradient given a number of parameters.
    empty       :: Int -> IO v
    -- | Clear a gradient (set all values to 0).
    clear       :: v -> IO ()
    -- | Fill gradient with a list of (position, number to add) pairs.
    -- The function may modify values of the given gradient.
    fill        :: [(Int, LogSigned)] -> v -> IO v
    -- | Contents of a gradient.  Zero-valued elements should not be reported.
    -- You can (and should!) use lazy IO here.  Or better, try some pipes
    -- implementation for that.
    content     :: v -> IO [(Int, LogSigned)]
    -- | Unions of gradients.  The function may modify values kept in
    -- a gradient supplied as the second argument.
    unionsTo    :: [v] -> v -> IO v

-- -- | Construct gradient from a list of (index, value) pairs.
-- -- All values from the list are added at respective gradient
-- -- positions.
-- fromList :: Grad v => [(Int, Double)] -> v
-- fromList = fromLogList . map (second logSigned)
-- {-# INLINE fromList #-}
-- 
-- -- | Collect gradient components with values in normal domain.
-- toList :: Grad v => v -> [(Int, Double)]
-- toList = map (second toNorm) . toLogList
-- {-# INLINE toList #-}
-- 
-- -- | Construct gradient from a list of (index, value) pairs.
-- -- All values from the list are added at respective gradient
-- -- positions.
-- fromListM :: MonadGrad m v => [(Int, Double)] -> m v
-- fromListM = fromLogListM . map (second logSigned)
-- {-# INLINE fromListM #-}
-- 
-- -- | Collect gradient components with values in normal domain.
-- toListM :: (Functor m, MonadGrad m v) => v -> m [(Int, Double)]
-- toListM = fmap (map (second toNorm)) . toLogListM
-- {-# INLINE toListM #-}
