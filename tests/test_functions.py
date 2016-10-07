import unittest

import numpy as np

from csxnet.util import cost_fns as cost, act_fns as act
from csxdata import etalon, roots, CData


class TestSoftmax(unittest.TestCase):

    def setUp(self):
        self.data = etalon()
        self.softmax = act["softmax"]
        self.rsmaxed = np.round(CData(roots["etalon"] + "smaxed.csv", cross_val=0.0).learning.astype(float), 4)

    def test_softmax_function(self):

        output = self.softmax(self.data.learning)
        output = np.round(output.astype(float), 4)

        self.assertTrue(np.allclose(output, self.rsmaxed))

    def test_softmax_derivative(self):
        gradients = self.softmax.derivative(self.rsmaxed)
        print(gradients)

if __name__ == '__main__':
    unittest.main()
