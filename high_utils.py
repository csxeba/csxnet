"""This module contains higher level library based utilities,
like SciPy, sklearn, Keras, Pillow etc."""


import numpy as np
from .nputils import ravel_to_matrix as rtm


def autoencode(X: np.ndarray, features: int or tuple, get_model: bool=False) -> np.ndarray:
    from keras.models import Sequential
    from keras.layers.core import Dense

    from csxnet.nputils import standardize

    def sanitize_params(ftrs):
        if isinstance(ftrs, int):
            ftrs = (ftrs, None)
        elif len(ftrs) < 2:
            ftrs = (ftrs[0], None)
        return ftrs

    def build_encoder(ftrs, dims):
        enc = Sequential()
        enc.add(Dense(input_dim=dims, output_dim=ftrs[0],
                      activation="tanh"))
        for ftr in ftrs[1:]:
            if ftr is None:
                break
            enc.add(Dense(output_dim=ftr, activation="tanh"))
        for ftr in ftrs[-1::-1]:
            if ftr is None:
                continue
            enc.add(Dense(output_dim=ftr, activation="tanh"))
        enc.add(Dense(output_dim=dims, activation="tanh"))
        enc.compile("adadelta", loss="mse")
        return enc

    print("Creating autoencoder model...")

    features = sanitize_params(features)

    data = standardize(rtm(X))
    dimensions = data.shape[1]

    encoder = build_encoder(features, dimensions)

    print("Training on data...")
    encoder.fit(data, data, batch_size=10, nb_epoch=30)

    weights, biases = encoder.layers[0].get_weights()
    params = [lay.get_weights() for lay in encoder.layers]
    if get_model:
        return params
    transformed = np.tanh(data.dot(weights) + biases)
    return transformed


def pca_transform(X: np.ndarray, factors: int, whiten: bool=False) -> np.ndarray:
    from sklearn.decomposition import PCA

    print("Fitting PCA...")
    pca = PCA(n_components=factors, whiten=whiten)
    data = pca.fit_transform(rtm(X))
    if data.shape[1] != factors and data.shape[1] == data.shape[0]:
        print("Warning! Couldn't calculate covariance matrix, used generalized inverse instead!")
    return data


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
