{-# LANGUAGE RecordWildCards #-}


module Numeric.SGD
  ( DataSet (..)
  , Config (..)
  , Dyna (..)
  , sgd
  ) where


import           Prelude hiding (elem)
import           Control.Monad (forM_)

import qualified System.Random as R

import qualified Data.IORef as IO

import qualified Numeric.SGD.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)


------------------------------- 
-- Data
-------------------------------


-- | Dataset stored on a disk
data DataSet elem = DataSet
  { size :: Int 
    -- ^ The size of the dataset; the individual indices are
    -- [0, 1, ..., size - 1]
  , elemAt :: Int -> IO elem
    -- ^ Get the dataset element with the given identifier
  }


-- | Top-level SGD static configuration
data Config = Config
  { iterNum :: Int
    -- ^ Number of iteration over the entire training dataset
  , batchSize :: Int
    -- ^ Size of the SGD batch
  , method :: Ada.Config -- Method
    -- ^ Selected SGD method
  , reportEvery :: Double
    -- ^ How often report the quality (with `1` meaning once per pass over the
    -- training data)
  }


-- -- | SGD method, together with the corresponding configuration
-- data Method
--   = AdaDelta Ada.Config


-- | Top-level SGD dynamic configuration
data Dyna p e = Dyna
  { gradient :: [e] -> p -> p
    -- ^ Net gradient on a particular dataset fragment
  , quality :: e -> p -> Double
    -- ^ Net quality measure w.r.t. the given dataset element.
    --
    -- NOTE: we assume that the quality on a dataset is the sum of the
    -- qualities on its individual elements
  }


------------------------------- 
-- SGD
-------------------------------


-- | Perform SGD.
sgd :: (ParamSet p) => Config -> Dyna p e -> DataSet e -> p -> IO p
sgd Config{..} Dyna{..} dataSet net0  = do
  Ada.adaDelta method dyna net0
  where
    dyna = Ada.Dyna
      { Ada.iterNum = ceiling
          $ fromIntegral (size dataSet * iterNum)
          / (fromIntegral batchSize :: Double)
      , Ada.gradient = \net -> do
          sample <- randomSample batchSize dataSet
          return $ gradient sample net
      , Ada.quality = \net -> do
          res <- IO.newIORef 0.0
          forM_ [0 .. size dataSet - 1] $ \ix -> do
            elem <- elemAt dataSet ix
            IO.modifyIORef' res (+ quality elem net)
          IO.readIORef res
      -- TODO: we could repot on a random sample!
      -- That could be also done more often!
      , Ada.reportEvery = ceiling
          $ fromIntegral (size dataSet) * reportEvery
          / fromIntegral batchSize
      }


------------------------------- 
-- Utils
-------------------------------


-- | Random dataset sample
randomSample :: Int -> DataSet a -> IO [a]
randomSample k dataSet
  | k <= 0 = return []
  | otherwise = do
      ix <- R.randomRIO (0, size dataSet - 1)
      x <- elemAt dataSet ix
      (x:) <$> randomSample (k-1) dataSet
