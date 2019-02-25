{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


module Numeric.SGD
  ( Config (..)
  , Method (..)
  , sgd
  , module Numeric.SGD.DataSet
  ) where


import           GHC.Generics (Generic)

import           Prelude hiding (elem)
import           Control.Monad (forM_)

import qualified Data.IORef as IO

import qualified Numeric.SGD.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import           Numeric.SGD.DataSet


------------------------------- 
-- Data
-------------------------------


-- | Top-level SGD static configuration
data Config = Config
  { iterNum :: Integer
    -- ^ Number of iteration over the entire training dataset
  , batchSize :: Integer
    -- ^ Size of the SGD batch
  , method :: Method
    -- ^ Selected SGD method
  , reportEvery :: Double
    -- ^ How often report the quality (with `1` meaning once per pass over the
    -- training data)
  } deriving (Show, Eq, Ord, Generic)


-- | SGD method, together with the corresponding configuration
data Method
  = AdaDelta
    { decay :: Double
      -- ^ Exponential decay parameter (see `Ada.decay`)
    , eps   :: Double
      -- ^ Epsilon value (see `Ada.eps`)
    }
  | Momentum
  deriving (Show, Eq, Ord, Generic)


-- -- | Top-level SGD dynamic configuration
-- data Dyna p e = Dyna
--   { gradient :: [e] -> p -> p
--     -- ^ Net gradient on a particular dataset fragment
--   , quality :: e -> p -> Double
--     -- ^ Net quality measure w.r.t. the given dataset element.
--     --
--     -- NOTE: we assume that the quality on a dataset is the sum of the
--     -- qualities on its individual elements
--   }


------------------------------- 
-- SGD
-------------------------------


-- | Perform SGD.
sgd
  :: (ParamSet p)
  => Config
  -> DataSet e
  -> ([e] -> p -> p)
    -- ^ Net gradient on a particular dataset fragment
  -> (e -> p -> Double)
    -- ^ Net quality measure w.r.t. the given dataset element (NOTE: we assume
    -- that the quality on a dataset is the sum of the qualities on its
    -- individual elements)
  -> p
    -- ^ Initial parameter values
  -> IO p
sgd Config{..} dataSet grad0 quality0 net0 =
  case method of
    AdaDelta{..} ->
      let cfg = Ada.Config
            { Ada.iterNum = ceiling
                $ fromIntegral (size dataSet) 
                * fromIntegral iterNum
                / (fromIntegral batchSize :: Double)
            , Ada.reportEvery = ceiling
                $ fromIntegral (size dataSet) * reportEvery
                / fromIntegral batchSize
            , Ada.decay = decay
            , Ada.eps = eps
            }
       in Ada.adaDelta cfg grad quality net0
    Momentum -> error "sgd: momentum not implemented yet!"
  where
    grad net = do
      sample <- randomSample (fromIntegral batchSize) dataSet
      return $ grad0 sample net
    quality net = do
      -- TODO: we could repot on a random sample!
      -- That could be also done more often!
      res <- IO.newIORef 0.0
      forM_ [0 .. size dataSet - 1] $ \ix -> do
        elem <- elemAt dataSet ix
        IO.modifyIORef' res (+ quality0 elem net)
      IO.readIORef res
