"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""


import numpy as np
from .nputils import ravel_to_matrix as rtm


def autoencode(X: np.ndarray, hiddens, validation: np.ndarray, epochs=5,
               get_model: bool=False) -> np.ndarray:
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.optimizers import Adadelta

    from csxnet.nputils import standardize

    def sanitize(ftrs):
        if isinstance(hiddens, int):
            ftrs = (hiddens,)
        return ftrs

    def build_encoder(hid, dims):
        enc = Sequential()
        enc.add(Dense(input_dim=dims, output_dim=hid[0],
                      activation="tanh"))
        if len(hid) > 1:
            for neurons in hid[1:]:
                enc.add(Dense(output_dim=neurons, activation="tanh"))
            for neurons in hid[-2:0:-1]:
                enc.add(Dense(output_dim=neurons, activation="tanh"))
        enc.add(Dense(output_dim=dims, activation="tanh"))
        enc.compile(Adadelta(), loss="mse")
        return enc

    print("Creating autoencoder model...")

    hiddens = sanitize(hiddens)
    data, mean, std = standardize(rtm(X), return_factors=True)
    dimensions = data.shape[1]

    encoder = build_encoder(hiddens, dimensions)
    print("Training on data...")
    if validation is not None:
        validation = standardize(validation, mean, std)
        validation = (validation, validation)
    encoder.fit(data, data, batch_size=20, nb_epoch=epochs, validation_data=validation)
    model = [layer.get_weights() for layer in encoder.layers]
    encoder, decoder = model[:len(hiddens)], model[len(hiddens):]

    transformed = np.tanh(data.dot(encoder[0][0]) + encoder[0][1])
    if len(encoder) > 1:
        for weights, biases in encoder[1:]:
            transformed = np.tanh(transformed.dot(weights) + biases)
    if get_model:
        return transformed, (encoder, decoder)
    else:
        return transformed


def pca_transform(X: np.ndarray, factors: int, whiten: bool=False,
                  get_model: bool=False) -> np.ndarray:
    from sklearn.decomposition import PCA

    print("Fitting PCA...")
    X = rtm(X)
    if factors is None:
        factors = X.shape[0]
        print("No factors is unspecified. Assuming all ({})!".format(factors))
    pca = PCA(n_components=factors, whiten=whiten)
    X = pca.fit_transform(X)
    if get_model:
        return X, pca
    else:
        return X


def plot(*lsts):
    import matplotlib.pyplot as plt
    for fn, lst in enumerate(lsts):
        plt.subplot(len(lsts), 1, fn + 1)
        plt.plot(lst)
    plt.show()


def image_to_array(imagepath):
    from PIL import Image
    return np.array(Image.open(imagepath))


def image_sequence_to_array(imageroot, outpath=None):
    import os

    flz = os.listdir(imageroot)

    print("Merging {} images to 3D array...".format(len(flz)))
    ar = np.stack([image_to_array(imageroot + image) for image in sorted(flz)])

    if outpath is not None:
        ar.dump(outpath)
        print("Images merged and dumped to {}".format(outpath))

    return ar


def th_haversine():
    """Returns a reference to the compiled haversine function!"""
    from theano import tensor as T
    from theano import function
    from csxnet.nputils import floatX

    coords1 = T.matrix("Coords1", dtype=floatX)
    coords2 = T.matrix("Coords2", dtype=floatX)

    R = np.array([6367], dtype="int32")  # Approximate radius of Mother Earth in kms
    coords1 = T.deg2rad(coords1)
    coords2 = T.deg2rad(coords2)
    lon1, lat1 = coords1[..., 0], coords1[..., 1]
    lon2, lat2 = coords2[..., 0], coords2[..., 1]
    dlon = lon1 - lon2
    dlat = lat1 - lat2
    d = T.sin(dlat / 2) ** 2 + T.cos(lat1) * T.cos(lat2) * T.sin(dlon / 2) ** 2
    e = 2 * T.arcsin(T.sqrt(d))
    d_haversine = e * R
    f_ = function([coords1, coords2], outputs=d_haversine)
    return f_
