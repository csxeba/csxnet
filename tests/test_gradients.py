import unittest

from csxdata import CData, roots
from csxdata.utilities.parsers import mnist_tolearningtable
from csxnet.ann import FeedForwardNet


class TestFeedForwardNet(unittest.TestCase):
    def setUp(self):
        self.data = CData(mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz", fold=False))

    def test_mse_with_sigmoid_output(self):
        self.net = FeedForwardNet(60, self.data, eta=1, cost="mse", activation="sigmoid",
                                  output_activation="sigmoid")
        self.net.fit(20, 1, 0)
        self.X, self.y = self.net.data.table("testing", m=10)

        relative_error, diff = self.net.gradient_check(self.X, self.y, fold=True)

        dfstr = "{:.2E}".format(relative_error)

        self.assertLessEqual(relative_error, 1e-2, "FATAL ERROR, {} (relerr) >= 1e-2".format(dfstr))
        self.assertLessEqual(relative_error, 1e-4, "ERROR, 1e-2 > {} (relerr) >= 1e-4".format(dfstr))
        self.assertLessEqual(relative_error, 1e-7, "SUSPICIOUS, 1e-4 > {} (relerr) >= 1e-7".format(dfstr))

    def test_xent_with_sigmoid_output(self):
        self.net = FeedForwardNet(60, self.data, eta=1, cost="xent", activation="sigmoid",
                                  output_activation="sigmoid")
        self.net.fit(20, 1, 0)
        self.X, self.y = self.net.data.table("testing", m=10)

        relative_error, diff = self.net.gradient_check(self.X, self.y, fold=True)

        dfstr = "{:.2E}".format(relative_error)

        self.assertLessEqual(relative_error, 1e-2, "FATAL ERROR, {} (relerr) >= 1e-2".format(dfstr))
        self.assertLessEqual(relative_error, 1e-4, "ERROR, 1e-2 > {} (relerr) >= 1e-4".format(dfstr))
        self.assertLessEqual(relative_error, 1e-7, "SUSPICIOUS, 1e-4 > {} (relerr) >= 1e-7".format(dfstr))

    def test_xent_with_softmax_output(self):
        self.net = FeedForwardNet(60, self.data, eta=1, cost="xent", activation="sigmoid",
                                  output_activation="softmax")
        self.net.fit(20, 1, 0)
        self.X, self.y = self.net.data.table("testing", m=10)

        relative_error, diff = self.net.gradient_check(self.X, self.y, fold=True)

        dfstr = "{:.2E}".format(relative_error)

        self.assertLessEqual(relative_error, 1e-2, "FATAL ERROR, {} (relerr) >= 1e-2".format(dfstr))
        self.assertLessEqual(relative_error, 1e-4, "ERROR, 1e-2 > {} (relerr) >= 1e-4".format(dfstr))
        self.assertLessEqual(relative_error, 1e-7, "SUSPICIOUS, 1e-4 > {} (relerr) >= 1e-7".format(dfstr))


if __name__ == '__main__':
    unittest.main()
