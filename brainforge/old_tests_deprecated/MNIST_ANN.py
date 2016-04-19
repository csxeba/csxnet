import time

from Architecture.Examples import *


def net1():
    eta = 0.5
    lmbda = 5.0
    cost = Xent
    epochs = 10

    data = CData("TestData/MNIST/mnist.pkl.gz", cross_val=.01)
    args = (data, eta, lmbda, cost)

    net = Network(*args)

    net.add_conv()
    net.add_pool()

    net.finalize_architecture()

    return net, epochs


def experiment():
    eta = 0.5
    lmbda = 5.0
    cost = Xent
    epochs = 100

    data = CData("TestData/MNIST/mnist.pkl.gz", cross_val=1-0.8571428571428571)

    net = Network(data, eta, lmbda, cost)
    net.finalize_architecture()

    for i in range(epochs):
        net.new_autoencode(100, 70)
        print("Round", i+1, ":", net.error)


def main():
    start = time.time()

    # net_definition = OnMNIST.by_Nielsen
    # net_definition = OnMNIST.shallow_CNN  # Too many test myData!
    # net_definition = OnMNIST.dropper
    net_definition = OnMNIST.by_Nielsen

    net, epochs = net_definition()

    # cn.new_autoencode(batches=5, batch_size=70)

    for epoch in range(epochs):
        print("------ EPOCH {} START -------".format(epoch+1))
        net.learn(batch_size=10)
        print("------ EPOCH {} DONE! -------".format(epoch+1))
        print("Tesing on T:", net.evaluate())
        print("Tesing on L:", net.evaluate("learning"))

    print("\n!!! END OF LEARNING !!!")
    print("Final scores:")
    print("T:", net.evaluate())
    print("L:", net.evaluate("learning"))
    print("Seconds elapsed:", int(time.time() - start))
    print("  Epoch average:", int((time.time() - start)/epochs))


if __name__ == '__main__':
    main()
