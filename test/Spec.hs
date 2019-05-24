{-# LANGUAGE NoMonomorphismRestriction #-}


import           Data.Ord (Ord(..))

import qualified Numeric.AD as AD

import           Test.Tasty (TestTree, testGroup)
import qualified Test.Tasty as Tasty
import qualified Test.Tasty.HUnit as U
import           Test.Tasty.HUnit ((@?=))
-- import qualified Test.Tasty.SmallCheck as SC

import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Adam as Adam
-- import qualified Numeric.SGD.AdaDelta as Ada


main :: IO ()
main = Tasty.defaultMain tests


tests :: TestTree
tests = testGroup "Tests" [unitTests]


unitTests = testGroup "Unit tests"
  [ U.testCase "Simple optimization" $ do
      U.assertBool "Momentum" . approxWith 0.01 4.18 $
        sgdWith (SGD.momentum SGD.def id)
      U.assertBool "Adam" . approxWith 0.01 4.18 $
        let cfg = SGD.def {Adam.alpha0 = 0.1}
         in sgdWith (SGD.adam cfg id)
      U.assertBool "AdaDelta" . approxWith 0.25 4.18 $
         sgdWith (SGD.adaDelta SGD.def id)
--   -- the following test does not hold
--   , U.testCase "List comparison (same length)" $
--       [1, 2, 3] `compare` [1,2,2] @?= LT
  ]


--------------------------------------------------
-- Main unit test
--------------------------------------------------


-- | The component objective functions
funs :: [Double -> Double]
funs =
  [ \x -> 0.3*x^2
  , \x -> -2*x
  , const 3
  , sin
  ]


-- | The corresponding derivatives
derivs :: [Double -> Double]
derivs =
  [ AD.diff $ \x -> 0.3*x^2
  , AD.diff $ \x -> -2*x
  , AD.diff $ const 3
  , AD.diff $ sin
  ]


-- | The total objective is the sum of the objective component functions
objective :: Double -> Double
objective x = sum $ map ($x) funs


-- | Perform SGD with the given SGD variant.
sgdWith typ = SGD.run typ (take 10000 $ cycle derivs) (0.0 :: Double)


--------------------------------------------------
-- Utils
--------------------------------------------------


-- | Is the second argument approximately equaly to the third one?
-- The first argument is the epsilon.
approxWith :: (Ord a, Floating a) => a -> a -> a -> Bool
approxWith eps x y =
  x >= y - eps && x <= y + eps
