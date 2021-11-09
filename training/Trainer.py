import numpy as np
from training.Model import Model
import training.Activation_funcs as aFunctions


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
        self.model = Model(layers)
        self.learning_rate = config['learning_rate']
        self.max_iter = config['iterations']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']
        self.losses = []
        self.features = []
        n_batches = self.data.shape[1] // self.batch_size
        self.batches = np.array_split(self.data[:, :self.batch_size * n_batches], n_batches, axis=1)
        # TODO: Finish
        x = 5

    @staticmethod
    def mean_squared_error(a_out, y):
        return ((a_out - y) ** 2).mean(axis=0)

    def train(self):
        x = self.data[:-1, 0:10]
        a_out = self.model.forward(x)
        y = x
        self.model.backward(a_out, y)

    # def run_epoch(self):
    #     x = np.random.rand(4)
    #     y = np.random.rand(4)
    #     self.plot_features.update(x, y)
    #     self.plot_weights.update_weight(0, 0, 0, x[0])
    #     self.plot_loss.update(x[0])
