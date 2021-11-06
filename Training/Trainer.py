from plots.PlotterFeatures import PlotterFeatures
from plots.PlotterLoss import PlotterLoss
from graphs.WeightsGraph import WeightsGraph
import numpy as np


class Trainer:
    def __init__(self, window):
        self.plot_features = PlotterFeatures(window, 'CANVAS_FEATURES')
        self.plot_weights = WeightsGraph(window['GRAPH_WEIGHTS'])
        self.plot_loss = PlotterLoss(window, 'CANVAS_LOSS')
        self.data = None

    def init_model(self, config):
        self.data = np.concatenate([np.append(
            np.random.multivariate_normal(mean, config['variances'][index], int(config['counts'][index])),
            np.full(int(config['counts'][index]), index)[:, np.newaxis], axis=1) for index, mean
                                    in enumerate(config['means'])], axis=0)

    def run_epoch(self):
        x = np.random.rand(4)
        y = np.random.rand(4)
        self.plot_features.update(x, y)
        self.plot_weights.update_weight(0, 0, 0, x[0])
        self.plot_loss.update(x[0])
