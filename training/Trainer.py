import numpy as np
from training.Model import Model
import training.Activation_funcs as aFunctions


def shuffle_data(d):
    idx = np.random.rand(d.shape[1]).argsort()
    return d[:, idx]


def normalize_data(d):
    d[0:4, :] /= np.max([np.abs(np.min(d[0:4, :])), np.abs(np.max(d[0:4, :]))])
    return d


class Trainer:
    def __init__(self, config):
        self.data = np.concatenate(
            [np.append(
                np.random.multivariate_normal(mean, config['variances'][index], int(config['counts'][index])),
                np.full(int(config['counts'][index]), index)[:, np.newaxis], axis=1) for index, mean in
                enumerate(config['means'])], axis=0).T
        self.data = shuffle_data(normalize_data(self.data))

        layers = [{"input_dim": int(layer), "output_dim": int(config['layers'][index + 1]),
                   "activation": getattr(aFunctions, config['activation']),
                   "derivative_a": getattr(aFunctions, f'{config["activation"]}_derivation')} for index, layer in
                  enumerate(config['layers'][:-1])]
        layers[-1]['activation'] = getattr(aFunctions, 'linear')
        layers[-1]['derivative_a'] = getattr(aFunctions, 'linear_derivation')
        self.learning_rate = config['learning_rate']
        self.model = Model(layers, self.learning_rate)
        self.max_iter = config['iterations']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']
        self.losses = []
        self.iteration = 0
        self.data_size = self.data.shape[1]
        self.batches = [self.data[:, i:i + self.batch_size] for i in range(0, self.data_size, self.batch_size)]
        self.n_batches = len(self.batches)
        self.features = [
            (np.append(self.model.forward(batch[:-1, :])[1], batch[-1:, :], axis=0), self.model.clean_mem())[0] for
            batch in self.batches]

    @staticmethod
    def mean_squared_error(a_out, y):
        return ((a_out - y) ** 2).mean(axis=0)

    def single_batch(self, index):
        batch = self.batches[index][:-1, :]
        a_out, features = self.model.forward(batch)
        self.features[index] = np.append(features, self.batches[index][-1:, :], axis=0)
        loss = self.mean_squared_error(a_out, batch)
        gradient = self.model.backward(a_out, batch)
        self.model.update(gradient, self.learning_rate)
        self.model.clean_mem()
        return np.mean(loss)

    def single_iteration(self):
        loss = 0
        for i in range(self.n_batches):
            loss += self.single_batch(i)
        self.iteration += 1
        return loss / self.n_batches

    def train(self):
        loss = []
        for i in range(self.max_iter):
            loss_iter = self.single_iteration()
            loss.append(loss_iter)
            if loss_iter < self.epsilon:
                return loss
        return loss
