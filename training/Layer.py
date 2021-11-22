import numpy as np


class Layer:
    """Single layer representation.
      Inspired by https://neuralnetworksanddeeplearning.com/chap2.html"""

    def __init__(self, layer_config):
        self.w = np.random.randn(layer_config["output_dim"], layer_config["input_dim"]) * 0.1
        self.b = np.random.randn(layer_config["output_dim"]) * 0.1
        self.a_func = layer_config["a_func"]
        self.a_prime = layer_config["a_prime"]
        self.a = None
        self.z = None

    def forward(self, x):
        self.z = np.dot(x, self.w.T) + self.b
        self.a = self.a_func(self.z)
        return self.a

    def backward(self, w_next, delta_next):
        ap = self.a_prime(self.z)
        delta_curr = np.dot(delta_next, w_next) * ap
        return delta_curr

    def update_state(self, w):
        self.w = w[:, 1:]
        self.b = w[:, 0]
