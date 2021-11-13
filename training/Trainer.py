import numpy as np
from training.Model import Model
import training.Activation_funcs as aFunctions


def normalize_data(d):
    d[:, :-1] /= np.max([np.abs(np.min(d[:, :-1])), np.abs(np.max(d[:, :-1]))])
    return d


class Trainer:
    def __init__(self, config):
        architecture = [{"input_dim": int(layer), "output_dim": int(config['layers'][index + 1]),
                         "a_func": getattr(aFunctions, config['a_func']),
                         "a_prime": getattr(aFunctions, f'{config["a_func"]}_prime')} for index, layer in
                        enumerate(config['layers'][:-1])]
        self.eta = config['learning_rate']
        self.model = Model(architecture)
        self.max_epochs = config['iterations']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']
        self.losses = []
        self.iteration = 0

        self.data = np.concatenate(
            [np.append(
                np.random.multivariate_normal(mean, config['variances'][index], int(config['counts'][index])),
                np.full(int(config['counts'][index]), index)[:, np.newaxis], axis=1) for index, mean in
                enumerate(config['means'])], axis=0)
        self.data = normalize_data(self.data)
        self.data_size = self.data.shape[0]
        self.batches = None
        self.batches_count = np.ceil(self.data_size / self.batch_size)
        self.shuffle_data()

        self.features = None
        self.update_features()

    def update_features(self):
        self.features = np.append(self.model.get_features(self.data[:, :-1]), self.data[:, -1:], axis=1)

    def shuffle_data(self):
        idx = np.random.rand(self.data.shape[0]).argsort()
        self.data = self.data[idx, :]
        self.batches = [self.data[k:k + self.batch_size, :] for k in range(0, self.data_size, self.batch_size)]

    @staticmethod
    def cost_derivative(a_out, y):
        return a_out - y

    @staticmethod
    def mean_squared_error(a_out, y):
        return (np.square(np.linalg.norm(a_out - y, axis=1))).mean(axis=0)

    def stochastic_gradient_descent(self):
        loss = []
        for e in range(self.max_epochs):
            loss_epoch = self.single_epoch()
            loss.append(loss_epoch)
        return loss

    def single_epoch(self):
        self.shuffle_data()
        loss_epoch = 0
        for batch in self.batches:
            loss_epoch += self.single_batch(batch)
        loss_epoch /= self.batches_count
        self.iteration += 1

        return loss_epoch

    def get_batch_by_index(self, index):
        return self.batches[index]

    def single_batch(self, batch):
        self.update_batch(batch[:, :-1])
        loss_batch = self.mean_squared_error(self.model.layers[-1].a, batch[:, :-1])
        return loss_batch

    def stochastic_gradient_descent_visualize(self, epochs):
        for e in range(epochs):
            self.shuffle_data()
            for batch in self.batches:
                self.update_batch(batch[:, :-1])
                print(f'Loss {self.mean_squared_error(self.model.layers[-1].a, batch[:, :-1])}')
                self.iteration += 1

    def update_batch(self, batch):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b, nabla_w = self.model.backpropagation(batch, batch, self.cost_derivative)
        for layer, nb, nw in zip(self.model.layers, nabla_b, nabla_w):
            layer.w = layer.w - self.eta * nw
            layer.b = layer.b - self.eta * nb
