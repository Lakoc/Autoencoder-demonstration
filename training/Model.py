import numpy as np


class Layer:
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

    def update(self, nabla_b, nabla_w):
        self.w += nabla_w
        self.b += nabla_b
        self.a = None
        self.z = None


class Model:
    def __init__(self, architecture):
        self.num_layers = len(architecture)
        self.layers = [Layer(layer_config) for layer_config in architecture]

    def feedforward(self, a):
        for layer in self.layers:
            a = layer.forward(a)

    def get_features(self, a):
        self.feedforward(a)
        return self.layers[self.num_layers // 2 - 1].a

    def get_weights_biases(self):
        weights = [layer.w for layer in self.layers]
        biases = [layer.b for layer in self.layers]
        return weights, biases

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
