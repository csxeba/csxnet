"""This module contains higher level library based utilities,
like SciPy, sklearn, Theano, Keras, etc."""


import numpy as np
from .nputils import ravel_to_matrix as rtm


def autoencode(X, features, get_model=False):
    from keras.models import Sequential
    from keras.layers.core import Dense

    from csxnet.nputils import standardize

    print("Creating autoencoder model...")

    data = standardize(rtm(X))
    dimensions = data.shape[1]

    encoder = Sequential()
    encoder.add(Dense(input_dim=dimensions, output_dim=features,
                      activation="tanh"))
    encoder.add(Dense(output_dim=dimensions, activation="tanh"))
    encoder.compile("adadelta", loss="mse")
    print("Training on data...")
    encoder.fit(data, data, batch_size=10, nb_epoch=30)
    weights, biases = encoder.layers[0].get_weights()
    if get_model:
        return weights, biases, "tanh"

    transformed = np.tanh(data.dot(weights) + biases)
    return transformed


def pca_transform(X: np.ndarray, factors, whiten=False):
    from sklearn.decomposition import PCA

    print("Fitting PCA...")
    pca = PCA(n_components=factors, whiten=whiten)
    data = pca.fit_transform(rtm(X))
    if data.shape[1] != factors and data.shape[1] == data.shape[0]:
        print("Warning! Couldn't calculate covariance matrix, used generalized inverse instead!")
    return data
