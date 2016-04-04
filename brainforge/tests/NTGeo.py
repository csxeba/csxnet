"""Stands for Nicotiana Tabacum Geographic [Analysis]..."""
import time

from Architecture.FFNN import FFLayerBrain
from Utility.DataModel import RData

start = time.time()

RATE = 2
HIDDENS = (15, 8)
REPEATS = 5000
ACTIVATION = "sigmoid"

myData = RData("TestData/NTGeo/Data.csv", 0.2, 2, True, sep=";", end="\n")

INPUTS, OUTPUTS = myData.neurons_required()
LAYOUT = [INPUTS] + list(HIDDENS) + [OUTPUTS]

fi = FFLayerBrain(eta=RATE, layout=LAYOUT)
print("Brain born with layout:", fi.layout)

for epoch in range(REPEATS):

    fi.learn(myData.table("learning"))
    if epoch % 100 == 0:
        print("Error @{}:".format(epoch),
              myData.test(fi), sep="\t")

print("Learning done in {} seconds. Final Error: {}"
      .format(round(time.time() - start, 2), myData.test(fi)))
