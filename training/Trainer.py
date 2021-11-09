import numpy as np
from training.Model import Model
import training.Activation_funcs as aFunctions


def shuffle_data(a):
    idx = np.random.rand(a.shape[1]).argsort()
    return a[:, idx]


class Trainer:
    def __init__(self, config):
        self.data = np.concatenate(
            [np.append(
                np.random.multivariate_normal(mean, config['variances'][index], int(config['counts'][index])),
                np.full(int(config['counts'][index]), index)[:, np.newaxis], axis=1) for index, mean in
                enumerate(config['means'])], axis=0).T
        layers = [{"input_dim": int(layer), "output_dim": int(config['layers'][index + 1]),
                   "activation": getattr(aFunctions, config['activation']),
                   "derivative_a": getattr(aFunctions, f'{config["activation"]}_derivation')} for index, layer in
                  enumerate(config['layers'][:-1])]
        self.data = shuffle_data(self.data)
        self.model = Model(layers)
        self.learning_rate = config['learning_rate']
        self.max_iter = config['iterations']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']
        self.losses = []
        self.features = []
        self.data_size = self.data.shape[1]
        self.batches = [self.data[:, i:i + self.batch_size] for i in range(0, self.data_size, self.batch_size)]
        self.n_batches = len(self.batches)

    @staticmethod
    def mean_squared_error(a_out, y):
        return ((a_out - y) ** 2).mean(axis=0)

    def single_batch(self, index):
        batch = self.batches[index][:-1, :]
        a_out, features = self.model.forward(batch)
        loss = self.mean_squared_error(a_out, batch)
        gradient = self.model.backward(a_out, batch)
        self.model.update(gradient, self.learning_rate)
        self.model.clean_mem()
        return np.mean(loss)

    def single_iteration(self):
        loss = 0
        for i in range(self.n_batches):
            loss += self.single_batch(i)
        return loss / self.n_batches

    def train(self):
        loss = []
        for i in range(self.max_iter):
            loss_iter = self.single_iteration()
            print(f'Current loss: {loss_iter}')
            loss.append(loss_iter)

    # def run_epoch(self):
    #     x = np.random.rand(4)
    #     y = np.random.rand(4)
    #     self.plot_features.update(x, y)
    #     self.plot_weights.update_weight(0, 0, 0, x[0])
    #     self.plot_loss.update(x[0])
