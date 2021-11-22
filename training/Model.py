import numpy as np
from training.Layer import Layer


class Model:
    """Class that maintains all layers and apply feedforward and backward propagation over layers.
     Inspired by https://neuralnetworksanddeeplearning.com/chap2.html"""

    def __init__(self, architecture, features_layer):
        self.num_layers = len(architecture)
        self.layers = [Layer(layer_config) for layer_config in architecture]
        # Skip first layer
        self.features_layer = features_layer - 1

    def get_shapes(self):
        return [(layer.w.shape[0], layer.w.shape[1] + 1) for layer in self.layers]

    def get_features(self, a):
        self.feedforward(a)
        return self.layers[self.features_layer].a

    def get_weights_biases(self):
        weights = [layer.w for layer in self.layers]
        biases = [layer.b for layer in self.layers]
        return weights, biases

    def feedforward(self, a):
        for layer in self.layers:
            a = layer.forward(a)

    def backpropagation(self, x, y, cost_derivative):
        nabla_b = [np.zeros(layer.b.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.w.shape) for layer in self.layers]

        self.feedforward(x)

        delta = cost_derivative(self.layers[-1].a, y) * self.layers[-1].a_prime(self.layers[-1].z)

        for i in range(1, self.num_layers):
            nabla_b[-i] = np.mean(delta, axis=0)
            nabla_w[-i] = np.dot(delta.T, self.layers[-i - 1].a)
            layer = self.layers[-i - 1]
            delta = layer.backward(self.layers[-i].w, delta)

        nabla_b[0] = np.mean(delta, axis=0)
        nabla_w[0] = np.dot(delta.T, x)

        return nabla_b, nabla_w
