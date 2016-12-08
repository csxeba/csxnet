from csxnet.brainforge.layers import DenseLayer
from csxnet.ann import Autoencoder

from csxdata import etalon


frame = etalon()
fanin, fanout = frame.neurons_required
model = Autoencoder(fanin, name="TestAutoEncoder")
model.add(DenseLayer(30, activation="sigmoid", trainable=0))
model.finalize("mse")

model.describe(1)

model.fit(frame.learning, batch_size=10, epochs=1, verbose=0)
if not model.gradient_check(frame.learning):
    raise RuntimeError("Gradient check failed!")

model.fit_csxdata(frame, monitor=["acc"])
