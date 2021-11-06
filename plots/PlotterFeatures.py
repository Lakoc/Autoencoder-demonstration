from plots.PlotterBase import Plot


class PlotterFeatures(Plot):
    def __init__(self, window, key):
        super(PlotterFeatures, self).__init__(window, key)

    def update(self, x, y):
        self.ax.cla()
        self.ax.scatter(x, y)
        self.fig_agg.draw()
