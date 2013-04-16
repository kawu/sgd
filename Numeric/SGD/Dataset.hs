{-# LANGUAGE RecordWildCards #-}


-- | Dataset abstraction.


module Numeric.SGD.Dataset
( 
-- * Dataset
  Dataset (..)
-- * Reading
, loadData
, sample
-- * Construction
, withVect
, withDisk
, withData
) where


import           Control.Monad (forM_)
import           Data.Binary (Binary, encodeFile, decodeFile)
import           System.IO.Unsafe (unsafeInterleaveIO)
import           System.IO.Temp (withTempDirectory)
import           System.FilePath ((</>))
import qualified System.Random as R
import qualified Data.Vector as V
import qualified Control.Monad.State.Strict as S


-- | A dataset with elements of type @a@.
data Dataset a = Dataset {
    -- | A size of the dataset.
      size      :: Int
    -- | Get dataset element with a given index.  The set of indices
    -- is of a {0, 1, .., size - 1} form.
    , elemAt    :: Int -> IO a }


-------------------------------------------
-- Reading
-------------------------------------------


-- | Lazily load dataset from a disk.
loadData :: Dataset a -> IO [a]
loadData Dataset{..} = lazyMapM elemAt [0 .. size - 1]


-- | A dataset sample of the given size.
sample :: R.RandomGen g => g -> Int -> Dataset a -> IO ([a], g)
sample g 0 _       = return ([], g)
sample g n dataset = do
    (xs, g') <- sample g (n-1) dataset
    let (i, g'') = R.next g'
    x <- dataset `elemAt` (i `mod` size dataset)
    return (x:xs, g'')


lazyMapM :: (a -> IO b) -> [a] -> IO [b]
lazyMapM f (x:xs) = do
    y <- f x
    ys <- unsafeInterleaveIO $ lazyMapM f xs
    return (y:ys)
lazyMapM _ [] = return []


-------------------------------------------
-- Construction
-------------------------------------------


-- | Construct dataset from a vector of elements and run the
-- given handler.
withVect :: [a] -> (Dataset a -> IO b) -> IO b
withVect xs handler =
    handler dataset
  where
    v = V.fromList xs
    dataset = Dataset
        { size      = V.length v
        , elemAt    = \k -> return (v V.! k) }


-- | Construct dataset from a list of elements, store it on a disk
-- and run the given handler.
withDisk :: Binary a => [a] -> (Dataset a -> IO b) -> IO b
withDisk xs handler = withTempDirectory "." ".sgd" $ \tmpDir -> do
    n <- flip S.execStateT 0 $ forM_ (zip xs [0 :: Int ..]) $ \(x, ix) -> do
        S.lift $ encodeFile (tmpDir </> show ix) x
        S.modify (+1)
    let at ix = decodeFile (tmpDir </> show ix)
    handler $ Dataset {size = n, elemAt = at}


-- | Use disk or vector dataset representation depending on
-- the first argument: when `True`, use `withDisk`, otherwise
-- use `withVect`.
withData :: Binary a => Bool -> [a] -> (Dataset a -> IO b) -> IO b
withData x = case x of
    True    -> withDisk
    False   -> withVect
