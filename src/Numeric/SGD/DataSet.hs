{-# LANGUAGE RecordWildCards #-}


-- | Provides the `DataSet` type which abstracts over the actual (IO-based)
-- representation of the training dataset.


module Numeric.SGD.DataSet
( 
-- * Dataset
  DataSet (..)
, shuffle
-- * Reading
, loadData
, randomSample
-- * Construction
, withVect
, withDisk
-- , withData
) where


import           Control.Monad (forM_)
import qualified Control.Monad.State.Strict as S

import           System.IO.Temp (withTempDirectory)
import           System.IO.Unsafe (unsafeInterleaveIO)
import           System.FilePath ((</>))
import qualified System.Random as R
import           System.Random.Shuffle (shuffleM)

import           Data.Binary (Binary, encodeFile, decode)
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as BL
import qualified Data.Vector as V
import qualified Data.Map.Strict as M


------------------------------- 
-- Type
-------------------------------


-- | Dataset stored on a disk
data DataSet elem = DataSet
  { size :: Int 
    -- ^ The size of the dataset; the individual indices are
    -- [0, 1, ..., size - 1]
  , elemAt :: Int -> IO elem
    -- ^ Get the dataset element with the given identifier
  }


-------------------------------------------
-- Reading
-------------------------------------------


-- | Lazily load the entire dataset from a disk.
loadData :: DataSet a -> IO [a]
loadData DataSet{..} = lazyMapM elemAt [0 .. size - 1]


-- -- | A dataset sample of the given size.
-- sample :: R.RandomGen g => g -> Int -> DataSet a -> IO ([a], g)
-- sample g 0 _       = return ([], g)
-- sample g n dataset = do
--     (xs, g') <- sample g (n-1) dataset
--     let (i, g'') = R.next g'
--     x <- dataset `elemAt` (i `mod` size dataset)
--     return (x:xs, g'')


-- | Shuffle the dataset.
shuffle :: DataSet a -> IO (DataSet a)
shuffle DataSet{..} = do
  let ixs = [0 .. size - 1]
  ixs' <- shuffleM ixs
  let m = M.fromList (zip ixs ixs')
  return $ DataSet
    { size = size
    , elemAt = elemAt . (m M.!)
    }


-- | Random dataset sample with a specified number of elements (loaded eagerly)
randomSample :: Int -> DataSet a -> IO [a]
randomSample k dataSet
  | k <= 0 = return []
  | otherwise = do
      ix <- R.randomRIO (0, size dataSet - 1)
      x <- elemAt dataSet ix
      (x:) <$> randomSample (k-1) dataSet


-------------------------------------------
-- Construction
-------------------------------------------


-- | Construct dataset from a list of elements, store it as a vector, and run
-- the given handler.
withVect :: [a] -> (DataSet a -> IO b) -> IO b
withVect xs handler =
    handler dataset
  where
    v = V.fromList xs
    dataset = DataSet
        { size      = V.length v
        , elemAt    = \k -> return (v V.! k) }


-- | Construct dataset from a list of elements, store it on a disk and run the
-- given handler.  Training elements must have the `Binary` instance for this
-- function to work.
withDisk :: Binary a => [a] -> (DataSet a -> IO b) -> IO b
withDisk xs handler = withTempDirectory "." ".sgd" $ \tmpDir -> do
    -- We use state monad to compute the number of dataset elements. 
    n <- flip S.execStateT 0 $ forM_ (zip xs [0 :: Int ..]) $ \(x, ix) -> do
        S.lift $ encodeFile (tmpDir </> show ix) x
        S.modify (+1)

    -- Avoid decodeFile laziness when using some older versions of the binary
    -- library (as of year 2019, this could be probably simplified)
    let at ix = do
          cs <- B.readFile (tmpDir </> show ix)
          return . decode $ BL.fromChunks [cs]

    handler $ DataSet {size = n, elemAt = at}


-------------------------------------------
-- Lazy IO Utils
-------------------------------------------


-- | Lazily evaluate each action in the sequence from left to right,
-- and collect the results.
lazySequence :: [IO a] -> IO [a]
lazySequence (mx:mxs) = do
    x   <- mx
    xs  <- unsafeInterleaveIO (lazySequence mxs)
    return (x : xs)
lazySequence [] = return []


-- | `lazyMapM` f is equivalent to `lazySequence` . `map` f.
lazyMapM :: (a -> IO b) -> [a] -> IO [b]
lazyMapM f = lazySequence . map f
