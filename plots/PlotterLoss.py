from plots.PlotterBase import Plot


class PlotterLoss(Plot):
    def __init__(self, window, key):
        super(PlotterLoss, self).__init__(window, key)
        self.loss = []

    def extend_loss(self, x):
        self.loss.extend(x)

    def update(self):
        self.ax.cla()
        self.ax.plot(self.loss)
        self.fig_agg.draw()

    def get_last_loss(self):
        return self.loss[-1] if len(self.loss) > 0 else 0

    def clear(self):
        self.loss = []
        self.ax.cla()
        self.fig_agg.draw()
