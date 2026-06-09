"""
Unit tests for curriculum cover-all epoch sampling in TrainDataset.
"""
import os
import sys
import tempfile
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), 'src')
sys.path.insert(0, src_dir)

from datasets import TrainDataset


class TestCurriculumCoverAllEpoch(unittest.TestCase):

    def _make_dataset(self, pool, tiles_per_image=3):
        tmp = tempfile.mkdtemp()
        train_annot = os.path.join(tmp, 'annotations', 'train')
        dataset = os.path.join(tmp, 'dataset')
        os.makedirs(train_annot)
        os.makedirs(dataset)
        ds = TrainDataset(train_annot, dataset, in_w=64, out_w=64, min_epoch_tiles=612)
        ds.set_allowed_fnames(pool)
        ds.set_curriculum_sampling(True, tiles_per_image)
        self.addCleanup(lambda: self._rmtree(tmp))
        return ds

    @staticmethod
    def _rmtree(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)

    def test_epoch_length_is_images_times_tiles(self):
        ds = self._make_dataset(['a.png', 'b.png'], tiles_per_image=3)
        self.assertEqual(len(ds), 6)

    def test_fname_for_index_maps_each_image_tiles_per_image_times(self):
        ds = self._make_dataset(['a.png', 'b.png'], tiles_per_image=3)
        self.assertEqual(ds.fname_for_index(0), 'a.png')
        self.assertEqual(ds.fname_for_index(1), 'a.png')
        self.assertEqual(ds.fname_for_index(2), 'a.png')
        self.assertEqual(ds.fname_for_index(3), 'b.png')
        self.assertEqual(ds.fname_for_index(4), 'b.png')
        self.assertEqual(ds.fname_for_index(5), 'b.png')

    def test_random_mode_unchanged_length_floor(self):
        tmp = tempfile.mkdtemp()
        train_annot = os.path.join(tmp, 'annotations', 'train')
        dataset = os.path.join(tmp, 'dataset')
        os.makedirs(train_annot)
        os.makedirs(dataset)
        ds = TrainDataset(train_annot, dataset, in_w=64, out_w=64, min_epoch_tiles=612)
        ds.set_allowed_fnames(['a.png', 'b.png'])
        ds.set_curriculum_sampling(False)
        self.assertEqual(len(ds), 612)
        self._rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
