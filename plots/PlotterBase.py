from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Plot(ABC):

    def __init__(self, window, key, figsize=None):
        fig, ax = plt.subplots(figsize=figsize)
        if figsize:
            fig.tight_layout()
        self.ax = ax
        self.fig_agg = self.draw_figure(window, key, fig)

    @staticmethod
    def draw_figure(window, key, figure):
        canvas_elem = window[key]
        canvas = canvas_elem.TKCanvas
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        return figure_canvas_agg

    @abstractmethod
    def update(self, *args):
        pass
