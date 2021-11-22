from plots.PlotterBase import Plot
import numpy as np
import matplotlib.pyplot as plt


class PlotterCMap(Plot):
    def __init__(self, window, key):
        super(PlotterCMap, self).__init__(window, key, figsize=(1, 4))
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient)).T

        self.ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('RdYlBu'), extent=[0, 1, -1.00001, 1.00001])
        self.ax.get_xaxis().set_visible(False)

        self.fig_agg.draw()

    def update(self):
        pass
