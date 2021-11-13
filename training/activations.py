import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))


def tanh(z):
    """The tanh function."""
    return np.tanh(z)


def tanh_prime(z):
    return 1 - tanh(z) ** 2
