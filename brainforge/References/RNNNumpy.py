"""
Learning RNN implementation from
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-\
rnn-with-python-numpy-and-theano/
"""

import numpy as np


class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        """
        :int word_dim: The size of the vocabulary
        :int hidden_dim: hidden layer size
        :int time_truncate: "explain cometh laitor.."
        """

        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate

        # These are 2D matrices of such and such size with uniformly distributed random entries
        # Resource refers to them as "network parameters"...
        self.U = np.random.uniform(-np.sqrt(1. / word_dim),
                                   np.sqrt(1. / word_dim),
                                   (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim),
                                   (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_dim),
                                   np.sqrt(1. / hidden_dim),
                                   (hidden_dim, hidden_dim))

    def think(self, x):
        """Predicting word probabilities"""
        # Total number of time steps
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))  # All hidden states are saved in s
        s[-1] = np.zeros(self.hidden_dim)  # We initialize s by adding a full-zero state
        o = np.zeros((T, self.word_dim))  # Store outputs here

        for t in np.arange(T):
            # So...
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = softmax(self.V.dot(s[t]))

        return [o, s]

    def predict(self, x):
        o, s = self.think(x)
        return np.argmax(o, axis=1)


def demo():
    np.random.seed(10)
    model = RNNNumpy(vocabulary_size)
