import unittest

from csxdata import etalon
from csxnet import Network


class TestRun(unittest.TestCase):

    def setUp(self):
        self.net = Network(etalon(), 0.03, 0.0, 0.0, 0.0, "xent", name="TestFFNN")
        self.net.add_fc(10)
        self.net.finalize_architecture()

    def test_architecture_is_right_after_initialization(self):
        print("; ".join(self.net.architecture))

    def testEpoch(self):
        self.net.fit(2, epochs=10, verbose=1)

    def testSoftMax(self):
        self.net.pop()
        self.net.finalize_architecture("softmax")
        self.net.fit(2, epochs=10, verbose=1)


if __name__ == '__main__':
    unittest.main()
