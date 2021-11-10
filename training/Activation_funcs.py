import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivation(da, z):
    sig = sigmoid(z)
    return da * sig * (1 - sig)


def linear(z):
    return z


def linear_derivation(da, z):
    return da


def tanh(z):
    return np.tanh(z)


def tanh_derivation(da, z):
    return da * (1 - tanh(z) ** 2)
