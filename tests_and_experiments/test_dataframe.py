import numpy as np


def test_categorical():
    from csxnet.utilities import roots
    from csxnet.data import CData, mnist_to_lt

    lt = mnist_to_lt(roots["misc"] + "mnist.pkl.gz", reshape=False)
    mnist = CData(lt)

    mnist.fit_autoencoder(60, epochs=1)
    assert mnist.learning.shape[-1] == mnist.testing.shape[-1] == 60,\
        "Unsuccessful autoencoding!\nlearning shape:\t{}\ntesting shape:\t{}".format(mnist.learning.shape,
                                                                                     mnist.testing.shape)
    print("Autoencoding was successful! Test passed!")
    mnist.reset_data(shuff=False, transform=None, params=None)

    mnist2 = CData(lt)

    assert all([i == j for i, j in zip(mnist.learning.shape, mnist2.learning.shape)]),\
        "Difference detected in data shapes"
    assert mnist.data.shape[-1] == 784,\
        "MNIST data lost its shape?"
    print("Shape tests passed on MNIST data!")
    sm1, sm2 = np.sum(mnist.data), np.sum(mnist2.data)
    assert sm1 == sm2,\
        "The sums of learning data differ by {}!\n{}\n{}".format(abs(sm1-sm2), sm1, sm2)
    print("Test of summation passed!")
    assert np.sum(mnist.data - lt[0]) == 0,\
        "Data core remained unchanged! Test of subtraction passed!"
    try:
        mnist2.data[0][0] = 2.0
    except ValueError:
        print("CData.data is read only. Test passed!")

    print("<<< CData: Every test passed! >>>")


if __name__ == '__main__':
    test_categorical()
