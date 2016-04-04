import numpy as np


class NeuralNetwork:
    def __init__(self, layers):
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

        self.number_of_layers = len(layers)

    def learn(self,
              training_data, test_data, number_of_epochs, batch_size, learning_rate):

        training_input_activations = training_data[0]
        training_expected_outputs = training_data[1]

        for i in range(1, number_of_epochs + 1):
            for k in range(0, len(training_input_activations[0]), batch_size):
                batch = \
                    (
                        training_input_activations[:, k:k + batch_size],
                        training_expected_outputs[:, k:k + batch_size]
                    )

                self.__learn_from_batch(batch, learning_rate)

            print("Progress: {0}. epoch {1}/{2} classified digits".format(
                i, self.__test(test_data), len(test_data[0][0])))

    def __learn_from_batch(self, batch, learning_rate):
        network_input_activations = batch[0]
        network_expected_outputs = batch[1]

        batch_weighted_outputs = []
        batch_activations = [network_input_activations]

        # feedforward
        for weight, bias in zip(self.weights, self.biases):
            weighted_outputs = \
                np.add(
                    np.dot(
                        weight,
                        batch_activations[-1]),
                    bias)

            batch_weighted_outputs.append(weighted_outputs)
            batch_activations.append(self.__sigmoid(weighted_outputs))

        # output error
        batch_errors = []
        batch_errors.append(
            np.multiply(
                np.subtract(
                    batch_activations[-1],
                    network_expected_outputs),
                self.__sigmoid_derivative(batch_weighted_outputs[-1])))

        # backward propagation
        for i in range(2, self.number_of_layers):
            batch_errors.append(
                np.multiply(
                    np.dot(
                        np.transpose(
                            self.weights[-i + 1]),
                        batch_errors[-i + 1]),
                    self.__sigmoid_derivative(batch_weighted_outputs[-i])))

        batch_errors.reverse()

        # gradient descent
        for i in range(self.number_of_layers - 1):
            self.biases[i] -= \
                np.multiply(
                    learning_rate,
                    np.mean(
                        batch_errors[i], axis=1, keepdims=True))

            self.weights[i] -= \
                np.multiply(
                    learning_rate,
                    np.divide(
                        np.dot(
                            batch_errors[i],
                            np.transpose(
                                batch_activations[i])),
                        len(batch[0][0])))

    def __sigmoid(self, z):
        return np.divide(1.0, np.add(1.0, np.exp(np.negative(z))))

    def __sigmoid_derivative(self, z):
        sigmoid_at_z = self.__sigmoid(z)
        return np.multiply(sigmoid_at_z, np.subtract(1, sigmoid_at_z))

    def __test(self, test_data):
        input_activations = test_data[0]
        expected_outputs = test_data[1]

        actual_outputs = self.__feedforward(input_activations)

        max_of_expected_outputs = np.argmax(expected_outputs, axis=0)
        max_of_actual_outputs = np.argmax(actual_outputs, axis=0)

        return np.sum(np.equal(max_of_expected_outputs, max_of_actual_outputs))

    def __feedforward(self, stims):
        activations = stims

        for weight, bias in zip(self.weights, self.biases):
            activations = \
                self.__sigmoid(
                    np.add(
                        np.dot(
                            weight,
                            activations),
                        bias))

        return activations
