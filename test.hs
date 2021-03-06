{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE NoMonomorphismRestriction #-}


import           Control.Arrow (first, second)
import qualified Numeric.AD as AD
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (BVar, W, Reifies)
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada


--------------------------------------------------
-- TEST 3
--------------------------------------------------


-- | The component objective functions
funs =
  [ \x -> 0.3*x^2
  , \x -> -2*x
  , const 3
  , sin
  ]


-- | The corresponding derivatives
derivs =
  map
  (\k -> AD.diff (funs !! k))
  [0 .. length funs - 1]
--   [ \x -> 0.6*x
--   , const (-2)
--   , const 0
--   , cos
--   ]


-- | The total objective is the sum of the objective component functions
objective x = sum $ map ($x) funs


main1 = print $
  SGD.run
    -- (SGD.momentum SGD.def id)
    -- (SGD.adaDelta (SGD.def {Ada.decay = 0.9, Ada.eps = 1.0e-8}) id)
    (SGD.adaDelta SGD.def id)
    -- (SGD.adam SGD.def id)
    (take 10000 $ cycle derivs)
    (0.0 :: Double)


-- --------------------------------------------------
-- -- TEST 2.2
-- --------------------------------------------------
-- 
-- 
-- -- | To bypass the impredicative polymorphism issue
-- newtype Fun = Fun { unFun :: forall a. Floating a => a -> a }
-- 
-- 
-- objectives :: [Fun]
-- objectives =
--   [ Fun $ \x -> 0.3*x^2
--   , Fun $ \x -> -2*x
--   , Fun $ const 3
--   , Fun sin
--   ]
-- 
-- 
-- -- | The total objective is the sum of the objective component functions
-- objective :: Double -> Double
-- objective x = sum $ map (($x) . unFun) objectives
-- 
-- 
-- -- | Gradient of a function for a given argument
-- grad :: Fun -> Double -> Double
-- grad (Fun f) x = BP.gradBP f x
-- 
-- 
-- main1 = print $
--   SGD.runSgd
--     (Mom.momentum Mom.def grad)
--     -- (Ada.adaDelta Ada.def grad)
--     (take 10000 $ cycle objectives)
--     0.0
-- 
-- 
-- --------------------------------------------------
-- -- TEST 2.1
-- --------------------------------------------------
-- 
-- 
-- objectives :: Floating a => [a -> a]
-- objectives =
--   [ \x -> 0.3*x^2
--   , \x -> -2*x
--   , const 3
--   , sin
--   ]
-- 
-- 
-- ixs :: [Int]
-- ixs = [0 .. length objectives - 1]
-- 
-- 
-- objective :: Floating a => a -> a
-- objective x = sum (map ($x) objectives)
-- 
-- 
-- grad :: Int -> Double -> Double
-- grad k = BP.gradBP (objectives !! k)
-- 
-- 
-- main1 = print $
--   SGD.runSgd
--     (Mom.momentum Mom.def grad)
--     -- (Ada.adaDelta Ada.def grad)
--     (take 10000 $ cycle ixs)
--     (3.0 :: Double)
-- 
-- 
-- main2 = do
--   p <- SGD.withDisk ixs $ \dataSet ->
--     SGD.runSgdIO
--       ( SGD.def
--         { SGD.iterNum = fromIntegral 10000
--         , SGD.reportEvery = 1000.0
--         }
--       )
--       (Mom.momentum Mom.def grad)
--       -- (Ada.adaDelta Ada.def grad)
--       (\k -> BP.evalBP (objectives !! k))
--       dataSet
--       (4.0 :: Double)
--   print p
-- 
-- 
-- --------------------------------------------------
-- -- TEST 1
-- --------------------------------------------------
-- 
-- 
-- objectiveBP (x, y) p =
--   let y' = (BP.auto x)^2 + p
--    in (BP.auto y - y') ^ 2
-- 
-- 
-- -- | Gradient on a training element
-- grad (x, y) =
--   BP.gradBP $ objectiveBP (x, y)
-- 
-- 
-- -- | Objective value on a training element
-- objective (x, y) =
--   BP.evalBP $ objectiveBP (x, y)
-- 
-- 
-- -- | Trainin dataset
-- trainData =
--   [(2.0, 3.0)]
-- 
-- 
-- main1 = print $
--   SGD.runSgd
--     (Mom.momentum Mom.def grad)
--     -- (Ada.adaDelta Ada.def grad)
--     (take 500 $ cycle trainData)
--     (1000.0 :: Double)
-- 
-- 
-- main2 = do
--   p <- SGD.withDisk trainData $ \dataSet ->
--     SGD.runSgdIO
--       ( SGD.def
--         { SGD.iterNum = fromIntegral 10000
--         , SGD.reportEvery = 1000.0
--         , SGD.batchRandom = False
--         }
--       )
--       -- (Mom.momentum Mom.def grad)
--       (Ada.adaDelta Ada.def grad)
--       objective
--       dataSet
--       (100.0 :: Double)
--   print p
