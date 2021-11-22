import numpy as np
import copy

"""Methods implemented in accordance with https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html"""


class NoOptimizer:
    """Simple optimizer just multiplying learning rate by nabla"""

    def __init__(self, config, layers):
        self.eta = config['eta']
        self.nw = [np.zeros(layer) for layer in layers]

    def update(self, epoch, layer_index, w, nw):
        self.nw[layer_index] = nw

    def update_weights(self, w, layer_index):
        return w - self.eta * self.nw[layer_index]


class Momentum(NoOptimizer):
    """Takes into account past gradients to smooth out the update"""

    def __init__(self, config, layers):
        super().__init__(config, layers)
        self.m_nw = copy.deepcopy(self.nw)
        self.beta1 = config['beta1']

    def update(self, epoch, layer_index, w, nw):
        self.m_nw[layer_index] = self.beta1 * self.m_nw[layer_index] + (1 - self.beta1) * nw
        self.nw[layer_index] = self.m_nw[layer_index]


class RMSProp(NoOptimizer):
    """Keeps an exponentially weighted average of the squares of past gradients"""

    def __init__(self, config, layers):
        super().__init__(config, layers)
        self.v_nw = [np.zeros(layer) for layer in layers]
        self.beta2 = config['beta2']
        self.epsilon = 1e-8

    def update(self, epoch, layer_index, w, nw):
        self.v_nw[layer_index] = self.beta2 * self.v_nw[layer_index] + (1 - self.beta2) * (nw ** 2)
        self.nw[layer_index] = nw / (np.sqrt(self.v_nw[layer_index]) + self.epsilon)


class Adam(Momentum, RMSProp):
    """Combines ideas from both RMSProp and Momentum."""
    def __init__(self, config, layers):
        super().__init__(config, layers)
        self.m_nw_corr = [np.zeros(layer) for layer in layers]
        self.v_nw_corr = [np.zeros(layer) for layer in layers]

    def update(self, epoch, layer_index, w, nw):
        Momentum.update(self, epoch, layer_index, w, nw)
        RMSProp.update(self, epoch, layer_index, w, nw)
        self.m_nw_corr[layer_index] = self.m_nw[layer_index] / (1 - self.beta1 ** epoch)
        self.v_nw_corr[layer_index] = self.v_nw[layer_index] / (1 - self.beta2 ** epoch)
        self.nw[layer_index] = (self.m_nw_corr[layer_index] / (np.sqrt(self.v_nw_corr[layer_index]) + self.epsilon))
