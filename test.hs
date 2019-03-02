{-# LANGUAGE FlexibleContexts #-}


import           Control.Arrow (first, second)
import qualified Numeric.Backprop as BP
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada


objectiveBP (x, y) p =
  let y' = (BP.auto x)^2 + p
   in (BP.auto y - y') ^ 2


-- | Gradient on a training element
grad (x, y) =
  BP.gradBP $ objectiveBP (x, y)


-- | Objective value on a training element
objective (x, y) =
  BP.evalBP $ objectiveBP (x, y)


-- | Trainin dataset
trainData =
  [(2.0, 3.0)]


main1 = print $
  SGD.runSgd
    (Mom.momentum Mom.def grad)
    -- (Ada.adaDelta Ada.def grad)
    (take 500 $ cycle trainData)
    (1000.0 :: Double)


main2 = do
  p <- SGD.withDisk trainData $ \dataSet ->
    SGD.runSgdIO
      ( SGD.def
        { SGD.iterNum = fromIntegral 10000
        , SGD.reportEvery = 1000.0
        , SGD.batchRandom = False
        }
      )
      -- (Mom.momentum Mom.def grad)
      (Ada.adaDelta Ada.def grad)
      objective
      dataSet
      (100.0 :: Double)
  print p
