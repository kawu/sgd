{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}


-- | Pure interface


module Numeric.SGD
  ( Method (..)
  , sgd
  , module Numeric.SGD.ParamSet
  ) where


import           GHC.Generics (Generic)

import           Data.Functor.Identity (runIdentity)

import qualified Pipes as P
import           Pipes ((>->))

import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Momentum as Mom
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.Pipe as P


-- | Available SGD methods, together with the corresponding configurations
data Method
  = AdaDelta {adaDeltaCfg :: Ada.Config}
  | Momentum {momentumCfg :: Mom.Config}
  deriving (Show, Eq, Ord, Generic)


-- | Stochastic gradient descent
sgd
  :: (ParamSet p)
  => Method
    -- ^ SGD method
  -> [e]
    -- ^ Training data stream
  -> (e -> p -> p)
    -- ^ Gradient on a training element
  -> p
    -- ^ Initial parameter values
  -> p
sgd method dataSet gradient p0 = runIdentity $
  P.result p0 
    (P.each dataSet >-> sgdPipe gradient p0)
  where
    sgdPipe =
      case method of
        Momentum cfg -> Mom.momentum cfg
        AdaDelta cfg -> Ada.adaDelta cfg


-- -- | Perform SGD.
-- sgd
--   :: (ParamSet p)
--   => Config
--   -> DataSet e
--   -> (e -> p -> p)
--     -- ^ Network gradient on a sample element
--   -> (e -> p -> Double)
--     -- ^ Value of the objective function on a sample element
--   -> p
--     -- ^ Initial parameter values
--   -> IO p
-- sgd Config{..} dataSet grad0 quality0 net0 = do
--   grad <-
--     if batchRandom
--        then pure randomGrad
--        else do
--          ixRef <- IO.newIORef 0 
--          pure (determGrad ixRef)
--   case method of
--     AdaDelta cfg ->
--       Ada.adaDeltaM cfg (args grad) net0
--     Momentum cfg ->
--       Mom.momentumM
--         (cfg {Mom.tau = iterScale (Mom.tau cfg)})
--         (args grad) net0
--   where
--     -- General SGD arguments
--     args grad = Args.Args
--       { Args.iterNum = realIterNum
--       , Args.gradient = grad
--       , Args.reportPeriod = realReportPeriod
--       , Args.report = report
--       }
--     -- Iteration scaling
--     iterScale x
--       = fromIntegral (size dataSet) * x
--       / (fromIntegral batchSize :: Double)
--     -- Number of iterations and reporting period
--     realIterNum = ceiling $ iterScale (fromIntegral iterNum)
--     realReportPeriod = ceiling $ iterScale reportPeriod
--     -- Gradient over a random sample 
--     randomGrad net = do
--       sample <- randomSample (fromIntegral batchSize) dataSet
--       return $ gradSample sample net
--     -- Gradient over a deterministically selected sample
--     determGrad ixRef net = do
--       p <- IO.readIORef ixRef
--       let q = p + fromIntegral batchSize - 1
--       sample <- range p q dataSet
--       IO.writeIORef ixRef $ p + 1 `mod` size dataSet
--       return $ gradSample sample net
--     -- Gradient on a training sample
--     gradSample sample net = do
--       let grads = map (\x -> grad0 x net) sample
--        in foldl' add (zero net) grads
--     -- Network quality over the entire training dataset
--     report net = do
--       putStr . show =<< quality net
--       putStrLn $ " (norm_2 = " ++ show (norm_2 net) ++ ")"
--     quality net = do
--       res <- IO.newIORef 0.0
--       forM_ [0 .. size dataSet - 1] $ \ix -> do
--         elem <- elemAt dataSet ix
--         IO.modifyIORef' res (+ quality0 elem net)
--       IO.readIORef res


-- -- | Dataset sample over a given (inclusive) range
-- range :: Int -> Int -> DataSet a -> IO [a]
-- range p q dataSet
--   | p > q = return []
--   | otherwise = do
--       x <- elemAt dataSet (p `mod` size dataSet)
--       (x:) <$> range (p+1) q dataSet
