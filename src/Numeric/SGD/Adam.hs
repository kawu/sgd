{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Provides the `adam` function which implements the Adam algorithm based on
-- the paper:
--
--     * https://arxiv.org/pdf/1412.6980


module Numeric.SGD.Adam
  ( Config(..)
  , adam
  ) where


import           GHC.Generics (Generic)

import           Prelude hiding (div)
-- import           Control.Monad (when)

import           Data.Default

import qualified Pipes as P

import           Numeric.SGD.Type
import           Numeric.SGD.ParamSet


-- | AdaDelta configuration
data Config = Config
  { alpha :: Double
    -- ^ Step size
  , beta1 :: Double
    -- ^ 1st exponential moment decay
  , beta2 :: Double
    -- ^ 1st exponential moment decay
  , eps   :: Double
    -- ^ Epsilon
  } deriving (Show, Eq, Ord, Generic)

instance Default Config where
  def = Config
    { alpha = 0.001
    , beta1 = 0.9
    , beta2 = 0.999
    , eps = 1.0e-8
    }


-- | Perform gradient descent using the Adam algorithm.  
-- See "Numeric.SGD.Adam" for more information.
adam
  :: (Monad m, ParamSet p)
  => Config
    -- ^ Adam configuration
  -> (e -> p -> p)
    -- ^ Gradient on a training element
  -> SGD m e p
adam Config{..} gradient net0 =

  let zr = zero net0 
   in go (1 :: Integer) zr zr net0

  where

    go t m v net = do
      x <- P.await
      let g = gradient x net
          m' = pmap (*beta1) m `add` pmap (*(1-beta1)) g
          v' = pmap (*beta2) v `add` pmap (*(1-beta2)) (g `mul` g)
          -- bias-corrected moment estimates 
          mb = pmap (/(1-beta1^t)) m'
          vb = pmap (/(1-beta2^t)) v'
          newNet = net `sub`
            ( pmap (*alpha) mb `div`
              (pmap (+eps) (pmap sqrt vb))
            )
      P.yield newNet
      go (t+1) m' v' newNet


-------------------------------
-- Utils
-------------------------------


-- -- | Scaling
-- scale :: ParamSet p => Double -> p -> p
-- scale x = pmap (*x)
-- {-# INLINE scale #-}
-- 
-- 
-- -- | Root square
-- squareRoot :: ParamSet p => p -> p
-- squareRoot = pmap sqrt
-- {-# INLINE squareRoot #-}
-- 
-- 
-- -- | Square
-- square :: ParamSet p => p -> p
-- square x = x `mul` x
-- {-# INLINE square #-}
