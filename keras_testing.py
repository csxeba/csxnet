import sys

from keras.models import Sequential
from keras.layers.core import Dense

from datamodel import mnist_to_lt
from datamodel import CData

dataroot = "D:/Data/" if sys.platform == "win32" else "/data/Prog/data/"
miscroot = dataroot + "misc/"
ltpath = miscroot + "mnist.pkl.gz"

mnist = CData(mnist_to_lt(ltpath, reshape=False))
mnist.standardize()

network = Sequential()
network.add(Dense(input_dim=784, output_dim=120, activation="tanh"))
network.add(Dense(input_dim=120, output_dim=10, activation="softmax"))
network.compile(optimizer="sgd", loss="categorical_crossentropy")

table = mnist.table("learning")
testing = mnist.table("testing")

network.fit(table[0], table[1], batch_size=20, nb_epoch=30, verbose=1, validation_data=testing)
