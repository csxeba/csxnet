import unittest

from csxdata import CData
from csxdata import roots
from numpy.linalg import norm

from csxnet.ann import Network
from csxnet.util import numerical_gradients, analytical_gradients, cost_fns as costs
from csxdata.utilities.parsers import mnist_tolearningtable


class TestNetwork(unittest.TestCase):

    def setUp(self):
        data = CData(mnist_tolearningtable(roots["misc"] + "mnist.pkl.gz", fold=False))
        data.transformation = "std"

        self.X, self.y = data.table("testing", m=20, shuff=False)

        self.net = Network(data, 1.0, 0.0, 0.0, 0.0, "mse", name="NumGradTestNetwork")
        self.net.add_fc(30)

    def test_mse_with_sigmoid_output(self):
        self.net.finalize("sigmoid")
        self.net.cost = costs.mse
        self.run_numerical_gradient_test()

    def test_xent_with_sigmoid_output(self):
        self.net.finalize_architecture("sigmoid")
        self.net.cost = costs.xent
        self.run_numerical_gradient_test()

    def test_xent_with_softmax_output(self):
        self.net.finalize_architecture("softmax")
        self.net.cost = costs.xent
        self.run_numerical_gradient_test()

    def run_numerical_gradient_test(self):
        self.net.fit(50, 1, 0)

        numerical = numerical_gradients(self.net, self.X, self.y)
        analytical = analytical_gradients(self.net, self.X, self.y)
        diff = analytical - numerical
        error = norm(diff) / max(norm(numerical), norm(analytical))

        dfstr = "{0: .4f}".format(error)

        self.assertLess(error, 1e-2, "FATAL ERROR, {} (relerr) >= 1e-2".format(dfstr))
        self.assertLess(error, 1e-4, "ERROR, 1e-2 > {} (relerr) >= 1e-4".format(dfstr))
        self.assertLess(error, 1e-7, "SUSPICIOUS, 1e-4 > {} (relerr) >= 1e-7".format(dfstr))


if __name__ == '__main__':
    unittest.main()
