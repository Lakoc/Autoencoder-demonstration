import PySimpleGUI as sg
import json
from gui.utils import normalize_config, validate_config
from training.Trainer import Trainer
from plots.PlotterStruct import Plotter
from gui.layout import init_layout


class Core:
    def __init__(self):
        sg.theme('Default1')
        self.window_1 = None
        self.window_2 = None
        self.config = json.load(open('config.json', 'r'))
        self.layout1, self.layout2 = init_layout(self.config)
        self.init_windows()
        self.trainer = None
        self.plotter = Plotter(self.window_2)
        self.win2_active = False
        self.batch_index = 0

    def init_windows(self):
        self.window_2 = sg.Window(
            "Autoencoder demonstration",
            self.layout2,
            location=(0, 0),
            finalize=True,
            element_justification="center",
            font="Helvetica 15",
        )
        self.window_2.hide()
        self.window_1 = sg.Window('Autoencoder config', self.layout1)

    def handle_window2(self):
        while True:
            event, values = self.window_2.read()

            if event == "Single Batch":
                loss = self.trainer.single_batch(self.trainer.get_batch_by_index(self.batch_index))
                self.batch_index += 1
                if self.batch_index == self.trainer.batches_count:
                    self.batch_index = 0
                    self.trainer.iteration += 1
                self.plotter.plot_loss.extend_loss([loss])
                self.update_window_state()

            if event == "Epoch":
                self.batch_index = 0
                self.process_single_epoch()

            if event == "100 epochs":
                self.batch_index = 0
                for _ in range(100):
                    self.process_single_epoch()

            if event == "Train":
                loss = self.trainer.stochastic_gradient_descent()
                self.plotter.plot_loss.extend_loss(loss)
                self.batch_index = 0
                self.update_window_state()

            if event == 'Edit config':
                self.win2_active = False
                self.window_1.un_hide()
                self.window_2.hide()
                break

            if event == sg.WIN_CLOSED:
                self.window_2.close()
                self.window_1.close()
                exit(0)

    def process_single_epoch(self):
        loss = self.trainer.single_epoch()
        self.plotter.plot_loss.extend_loss([loss])
        self.update_window_state()

    def update_window_state(self):
        weights, biases = self.trainer.model.get_weights_biases()
        self.window_2['Counter'].update(
            f"Batch: {self.batch_index}\nEpoch: {self.trainer.iteration}\n"
            f"Current loss: {self.plotter.plot_loss.get_last_loss():.3f}")
        self.trainer.update_features()
        self.plotter.plot_features.update(self.trainer.features)
        self.plotter.plot_weights.update(weights, biases)
        self.plotter.plot_loss.update()

    def init_plots(self):
        weights, biases = self.trainer.model.get_weights_biases()
        self.plotter.plot_weights.init_model(weights)
        self.plotter.plot_loss.clear()
        self.batch_index = 0
        self.trainer.iteration = 0
        self.update_window_state()

    def move_to_win2(self, config):
        config = normalize_config(config)

        errors = validate_config(config)
        if len(errors) > 0:
            sg.popup_error(f'Invalid config.\n {errors}')
        else:
            self.trainer = Trainer(config)
            self.init_plots()

            self.win2_active = True
            self.window_2.un_hide()
            self.window_1.hide()

            self.handle_window2()

    def run(self):
        while True:
            event, config = self.window_1.read()

            if event == sg.WIN_CLOSED:
                break

            if event == 'Done' and not self.win2_active:
                self.move_to_win2(config)

        self.window_1.close()
        self.window_2.close()
