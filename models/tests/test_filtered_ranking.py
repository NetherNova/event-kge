import unittest
from models.model import ranking_error_triples
import numpy as np


class FilteredRankingTestCase(unittest.TestCase):
    def setUp(self):
        self.left = range(3)
        self.rel = np.zeros((3), dtype=np.int32)
        self.right = np.zeros((3), dtype=np.int32)
        self.scores_l = np.array([[-1, -2, -3],
                                  [-3, -2, -1], # the third score (which is the best) in this list should go to -inf, thus second one is correct
                                  [-2, -2, -3]], # in this case, validation and test are the same, i.e. -inf is the correct entity
                                 dtype=np.float32)
        self.scores_r = np.random.random((3, 3))
        self.filtered = [(2, 0, 0)]

    def test_something(self):
        errl, errr = ranking_error_triples(self.filtered, self.scores_l, self.scores_r,
                                           self.left, self.rel, self.right)
        self.assertEqual(errl[1], 1)
        self.assertEqual(errl[0], 1)
        self.assertEqual(errl[2], 3)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
