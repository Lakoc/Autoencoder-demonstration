import numpy as np

from plots.PlotterBase import Plot


class PlotterFeatures(Plot):
    def __init__(self, window, key):
        super(PlotterFeatures, self).__init__(window, key)

    def update(self, features):
        self.ax.cla()
        features = np.concatenate(features, axis=1)
        x = features[0, :]
        y = features[1, :]
        c = features[2, :]
        self.ax.scatter(x, y, c=c)
        self.fig_agg.draw()
