from Architecture.NNModel import Network
from Utility.DataModel import CData
from Utility.cost import *


ETA = 0.1
LAMBDA = 0.3
COST = Xent
EPOCHS = 600
CROSSVAL = 0.50


def construct(data):
    net = Network(data, ETA, LAMBDA, COST)

    net.add_fc(120)
    net.finalize_architecture()

    return net


def experiment(verbose=1):
    myData = CData("TestData/Dohany_ANN/full.csv", cross_val=CROSSVAL, header=True, sep=";")

    # archon = FFLayerBrain(hiddens=hiddens, myData=myData, eta=eta, cost=MSE)
    archon = construct(myData)

    accuracies = []

    for epoch in range(1, EPOCHS+1):
        archon.learn(batch_size=int(archon.N/2))
        accuracies.append((archon.evaluate("testing"), archon.evaluate("learning")))

        # if epoch % 100 == 0:
        #     print("Epoch: {}".format(epoch))
        #     print("Accuracy on T:", accuracies[-1][0])
        #     print("Accuracy on L:", accuracies[-1][1])

    print("Epoch: {}".format(epoch))
    print("Accuracy on T:", accuracies[-1][0])
    print("Accuracy on L:", accuracies[-1][1])

    return accuracies


def plot(accuracies):
    import matplotlib.pyplot as plt
    accuracies = list(zip(*accuracies))
    plt.plot([0] + list(accuracies[0]), label="T")
    plt.plot([0] + list(accuracies[1]), label="L")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0.0, 1.0))
    plt.show()


if __name__ == '__main__':
    plot(experiment(verbose=0))

