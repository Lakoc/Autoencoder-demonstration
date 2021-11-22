import numpy as np
from training.Model import Model
import training.activations as a_functions
from training import optimizers


def normalize_data(d):
    d[:, :-1] /= np.max([np.abs(np.min(d[:, :-1])), np.abs(np.max(d[:, :-1]))])
    return d


def append_bias_weight(b, w):
    return np.append(b[:, np.newaxis], w, axis=1)


class Trainer:
    def __init__(self, config):
        """Maintains model training"""
        architecture = [{"input_dim": int(layer), "output_dim": int(config['layers'][index + 1]),
                         "a_func": getattr(a_functions, config['a_func']),
                         "a_prime": getattr(a_functions, f'{config["a_func"]}_prime')} for index, layer in
                        enumerate(config['layers'][:-1])]

        self.model = Model(architecture, config['layer'])
        self.optimizer = getattr(optimizers, config['optimizer'])(config, self.model.get_shapes())

        self.losses = []
        self.iteration = 0

        self.max_epochs = config['iterations']
        self.batch_size = config['batch_size']
        self.epsilon = config['epsilon']

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

    @staticmethod
    def cost_derivative(a_out, y):
        return a_out - y

    @staticmethod
    def mean_squared_error(a_out, y):
        return (np.square(np.linalg.norm(a_out - y, axis=1))).mean(axis=0)

    def update_features(self):
        """Features are data representation in 2D space"""
        self.features = np.append(self.model.get_features(self.data[:, :-1], ), self.data[:, -1:], axis=1)

    def shuffle_data(self):
        """Shuffle data by indexes"""
        idx = np.random.rand(self.data.shape[0]).argsort()
        self.data = self.data[idx, :]
        self.batches = [self.data[k:k + self.batch_size, :] for k in range(0, self.data_size, self.batch_size)]

    def get_batch_by_index(self, index):
        return self.batches[index]

    def stochastic_gradient_descent(self):
        loss = []
        for e in range(self.max_epochs):
            loss_epoch = self.single_epoch()
            loss.append(loss_epoch)
            if loss_epoch < self.epsilon:
                return loss
        return loss

    def single_epoch(self):
        self.shuffle_data()
        loss_epoch = 0
        for batch in self.batches:
            loss_epoch += self.single_batch(batch)
        loss_epoch /= self.batches_count
        self.iteration += 1

        return loss_epoch

    def single_batch(self, batch):
        self.update_by_batch(batch[:, :-1])
        loss_batch = self.mean_squared_error(self.model.layers[-1].a, batch[:, :-1])
        return loss_batch

    def update_by_batch(self, batch):
        """Update model state by single batch using selected optimizer"""
        nabla_b, nabla_w = self.model.backpropagation(batch, batch, self.cost_derivative)
        epoch = self.iteration + 1
        for index, (layer, nb, nw) in enumerate(zip(self.model.layers, nabla_b, nabla_w)):
            w = append_bias_weight(layer.b, layer.w)
            self.optimizer.update(epoch, index, w,
                                  append_bias_weight(nb, nw))
            layer.update_state(self.optimizer.update_weights(w, index))
