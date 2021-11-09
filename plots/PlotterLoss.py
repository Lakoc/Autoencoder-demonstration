from plots.PlotterBase import Plot


class PlotterLoss(Plot):
    def __init__(self, window, key):
        super(PlotterLoss, self).__init__(window, key)
        self.loss = []

    def update(self, x):
        self.loss.append(x)
        self.ax.cla()
        self.ax.plot(self.loss)
        self.fig_agg.draw()