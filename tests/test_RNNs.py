import unittest

from csxnet import Network
from csxdata import etalon


class TestRNN(unittest.TestCase):

    def setUp(self):
        model = Network(etalon(), 0.01, 0.0, 0.0, 0.0, "xent", "TestRNN")
        model.add_rec(3)
