import numpy as np


class Model:
    def __init__(self, architecture):
        self.weights = [np.random.randn(
            layer["output_dim"], layer["input_dim"]) * 0.1 for idx, layer in enumerate(architecture)]
        self.biases = [np.random.randn(
            layer["output_dim"], 1) * 0.1 for idx, layer in enumerate(architecture)]
        self.architecture = architecture
        self.memory = {'a': [], 'z': []}
        self.n_layers = len(self.weights)

    @staticmethod
    def der_mean_squared_error(a_out, y):
        return ((a_out - y) * 2).mean(axis=0)

    @staticmethod
    def single_layer_forward_propagation(a_prev, w_curr, b_curr, activation_func):
        z_curr = np.dot(w_curr, a_prev) + b_curr
        return activation_func(z_curr), z_curr

    @staticmethod
    def single_layer_backward_propagation(da_curr, w_curr, z_curr, a_prev, activation_der):
        # number of examples
        m = a_prev.shape[1]

        # calculation of the activation function derivative
        dz_curr = activation_der(da_curr, z_curr)

        # derivative of the matrix W
        dw_curr = np.dot(dz_curr, a_prev.T) / m
        db_curr = np.sum(dz_curr, axis=1, keepdims=True) / m

        # derivative of the matrix A_prev
        da_prev = np.dot(w_curr.T, dz_curr)

        return da_prev, dw_curr, db_curr

    def clean_mem(self):
        self.memory = {'a': [], 'z': []}

    def forward(self, x):
        a_curr = x

        for index, weights in enumerate(self.weights):
            a_prev = a_curr

            a_curr, z_curr = Model.single_layer_forward_propagation(a_prev, weights, self.biases[index],
                                                                    self.architecture[index]['activation'])

            self.memory['a'].append(a_prev)
            self.memory['z'].append(z_curr)

        return a_curr, self.memory['a'][self.n_layers // 2]

    def backward(self, a_out, y):

        # initiation of gradient descent algorithm
        da_prev = Model.der_mean_squared_error(a_out, y)
        gradients = {'b': [], 'w': []}

        for index in range(self.n_layers):
            # we number network layers from 1

            da_curr = da_prev

            a_prev = self.memory['a'][-index]
            z_curr = self.memory['z'][-index]
            w_curr = self.weights[-index]

            der_activation = self.architecture[-index]['derivative_a']
            da_prev, dw_curr, db_curr = self.single_layer_backward_propagation(
                da_curr, w_curr, z_curr, a_prev, der_activation)

            gradients['w'].append(dw_curr)
            gradients['b'].append(db_curr)
        return gradients

    def update(self, gradients, learning_rate):
        for layer_idx in range(self.n_layers):
            self.weights[layer_idx] -= learning_rate * gradients['w'][-layer_idx]
            self.biases[layer_idx] -= learning_rate * gradients['b'][-layer_idx]
