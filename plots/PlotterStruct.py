from plots.PlotterFeatures import PlotterFeatures
from plots.PlotterLoss import PlotterLoss
from graphs.WeightsGraph import WeightsGraph
from plots.PlotterCMap import PlotterCMap


class Plotter:
    def __init__(self, window):
        self.plot_features = PlotterFeatures(window, 'CANVAS_FEATURES')
        self.plot_weights = WeightsGraph(window['GRAPH_WEIGHTS'])
        self.plot_loss = PlotterLoss(window, 'CANVAS_LOSS')
        self.plot_c_map = PlotterCMap(window, 'CANVAS_CMAP')
