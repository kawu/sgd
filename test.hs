import           Control.Arrow (first, second)
import qualified Numeric.Backprop as BP
import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada


-- | Gradient on a training element
grad (x, y) = BP.gradBP $ \p ->
  let y' = (BP.auto x)^2 + p
   in (BP.auto y - y') ^ 2


-- | Trainin dataset
trainData =
  [(2.0, 3.0)]


main = print $
  SGD.runSgd
    (Mom.momentum Mom.def grad)
    (take 1000 $ cycle trainData)
    (20.0 :: Double)
