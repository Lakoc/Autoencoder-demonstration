import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivation(da, z):
    sig = sigmoid(z)
    return da * sig * (1 - sig)
