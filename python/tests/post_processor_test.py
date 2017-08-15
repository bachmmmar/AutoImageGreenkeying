import unittest
from processors import *

class PostProcessorTest(unittest.TestCase):

    def test_calculate_rotated_dimension_30degree(self):
        cut = PostProcessor()

        size = cut.calculate_rotated_dimension(12,15,30)
        self.assertAlmostEquals(size[0], 18)
        self.assertAlmostEquals(size[1], 19)

    def test_calculate_rotated_dimension_5degree(self):
        cut = PostProcessor()

        size = cut.calculate_rotated_dimension(700,1200,5)
        self.assertAlmostEquals(size[0], 802)
        self.assertAlmostEquals(size[1], 1257)
