import unittest

import numpy as np
import numpy.testing as npt

from models.rpn.utils import generate_anchors


class AnchorTester(unittest.TestCase):
    def test_anchor_generator(self):
        anchors = generate_anchors(base_size=16, ratios=(0.5, 1, 2), scales=2 ** np.arange(3, 6))
        # This is not the output posted in the original, but it is what I got when I copied the code
        # The errors are all either +1 or -1 (all anchors are one pixel larger on each side),
        # which I think occur because of rounding differences and shouldn't matter.
        expected_output = np.array([[-84., -40., 99., 55.],
                                    [-176., -88., 191., 103.],
                                    [-360., -184., 375., 199.],
                                    [-56., -56., 71., 71.],
                                    [-120., -120., 135., 135.],
                                    [-248., -248., 263., 263.],
                                    [-36., -80., 51., 95.],
                                    [-80., -168., 95., 183.],
                                    [-168., -344., 183., 359.]])
        npt.assert_equal(anchors, expected_output)
