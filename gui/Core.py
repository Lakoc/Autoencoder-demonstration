import PySimpleGUI as sg
import numpy as np
from training.Trainer import Trainer
import json
from plots.Plotter import Plotter


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def remove_values_from_list_to_float(the_list, val):
    return [float(value) for value in the_list if value != val]


def load_3d_arr_from_string(arr):
    arr = arr.replace('[', '').replace(']', '').split('\n')
    count = arr.count('') + 1

    arr = remove_values_from_list(arr, '')
    group_size = len(arr) // count
    groups = [remove_values_from_list_to_float(val.split(' '), '') for group in range(count) for val in
              arr[group * group_size: (group + 1) * group_size]]
    groups = [groups[group * group_size: (group + 1) * group_size] for group in range(count)]
    return np.array(groups)


def normalize_config(config):
    config['variances'] = load_3d_arr_from_string(config['variances'])
    config['means'] = load_3d_arr_from_string(config['means'])[0, :]
    config['counts'] = load_3d_arr_from_string(config['counts'])[0, 0, :]
    config['layers'] = load_3d_arr_from_string(config['layers'])[0, 0, :]
    config['batch_size'] = int(config['batch_size'])
    config['iterations'] = int(config['iterations'])
    config['epsilon'] = float(config['epsilon'])
    config['learning_rate'] = float(config['learning_rate'])
    config['activation'] = config['activation'][0]
    return config


class Core:
    def __init__(self):
        sg.theme('Default1')
        self.layout1 = None
        self.layout2 = None
        self.window_1 = None
        self.window_2 = None
        self.config = json.load(open('config.json', 'r'))
        self.init_layout()
        self.init_windows()
        self.trainer = None
        self.plotter = Plotter(self.window_2)
        self.win2_active = False
        self.batch_index = 0

    def init_layout(self):
        activation_choices = ('tanh', 'sigmoid')

        self.layout1 = [[sg.Text('Data config (ensure to have corresponding dimensionality of inputs) :\n')],
                        [sg.Text('Clusters (please ensure to bracket each cluster data):')],
                        [sg.Text('Cluster means:')], [
                            sg.Multiline(np.array(self.config['data']['means']), size=(80, 5), key='means')],
                        [sg.Text('Cluster variances:')],
                        [sg.Multiline(
                            np.array(self.config['data']['variances']),
                            size=(80, 10), key='variances')],
                        [sg.Text('Clusters number of points:')],
                        [sg.Multiline(np.array(self.config['data']['counts']), size=(80, 5), key='counts')],
                        [sg.Text('\nModel config:\n')],
                        [sg.Text(
                            'Number of neurons in layers\n(count must be odd, middle value must be equal to 2, first and last layer neuron count must equal):')],
                        [sg.Multiline(np.array(self.config['model']['layers']), size=(80, 5), key='layers')],
                        [sg.Text('Choose activation function, used over layers:')],
                        [sg.Listbox(activation_choices, size=(15, len(activation_choices)), default_values=['tanh'],
                                    key='activation')],
                        [sg.Text('\nTraining config:\n')],
                        [sg.Text('Batch size (1 - number of generated points):'), sg.Input(32, key='batch_size')],
                        [sg.Text('Max iterations:'), sg.Input(self.config['training']['iterations'], key='iterations')],
                        [sg.Text('Epsilon:'),
                         sg.Input(self.config['training']['epsilon'], key='epsilon')],
                        [sg.Text('Learning rate:'),
                         sg.Input(self.config['training']['learning_rate'], key='learning_rate')],
                        [sg.Button('Done')]]
        self.layout2 = [
            [sg.Text("Batch: 0\nEpoch: 0", key='Counter', expand_x=True), sg.Button("Edit config")],
            [sg.Canvas(key="CANVAS_FEATURES", expand_x=True, size=(400, 400)),
             sg.Canvas(key="CANVAS_LOSS", size=(400, 400))],
            [sg.Graph(canvas_size=(800, 400), graph_bottom_left=(0, 0), graph_top_right=(800, 400),
                      background_color='white',
                      key='GRAPH_WEIGHTS')],
            [sg.Button("Single Batch"), sg.Button("Epoch"), sg.Button("100 epochs"), sg.Button("Train")],
        ]

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
                loss = self.trainer.single_batch(self.batch_index)

                self.batch_index += 1
                if self.batch_index == self.trainer.n_batches:
                    self.batch_index = 0
                    self.trainer.iteration += 1
                self.window_2['Counter'].update(f"Batch: {self.batch_index}\nEpoch: {self.trainer.iteration}")
                self.plotter.plot_loss.update([loss])
                weights, biases = self.trainer.model.weights, self.trainer.model.biases
                self.plotter.plot_weights.update(weights, biases)
                self.plotter.plot_features.update(self.trainer.features)

            if event == "Epoch":
                loss = self.trainer.single_iteration()
                self.plotter.plot_loss.update([loss])
                self.plotter.plot_features.update(self.trainer.features)
                weights, biases = self.trainer.model.weights, self.trainer.model.biases
                self.plotter.plot_weights.update(weights, biases)
                self.batch_index = 0
                self.window_2['Counter'].update(f"Batch: {self.batch_index}\nEpoch: {self.trainer.iteration}")

            if event == "100 epochs":
                self.batch_index = 0
                for _ in range(100):
                    loss = self.trainer.single_iteration()
                    self.plotter.plot_loss.update([loss])
                    self.plotter.plot_features.update(self.trainer.features)
                    weights, biases = self.trainer.model.weights, self.trainer.model.biases
                    self.plotter.plot_weights.update(weights, biases)
                    self.window_2['Counter'].update(f"Batch: {self.batch_index}\nEpoch: {self.trainer.iteration}")

            if event == "Train":
                loss = self.trainer.train()
                self.plotter.plot_loss.update(loss)
                self.plotter.plot_features.update(self.trainer.features)
                weights, biases = self.trainer.model.weights, self.trainer.model.biases
                self.plotter.plot_weights.update(weights, biases)
                self.batch_index = 0
                self.window_2['Counter'].update(f"Batch: {self.batch_index}\nEpoch: {self.trainer.iteration}")

            if event == 'Edit config':
                self.win2_active = False
                self.window_1.un_hide()
                self.window_2.hide()
                break

            if event == sg.WIN_CLOSED:
                self.window_2.close()
                self.window_1.close()
                exit(0)

    def run(self):

        while True:
            event, config = self.window_1.read()

            if event == sg.WIN_CLOSED:
                break

            if event == 'Done' and not self.win2_active:
                config = normalize_config(config)
                self.trainer = Trainer(config)
                self.plotter.plot_weights.init_model(self.trainer.model.weights)
                self.plotter.plot_loss.clear()
                self.plotter.plot_features.update(self.trainer.features)
                self.window_2['Counter'].update(f"Batch: 0\nEpoch: 0")

                self.win2_active = True
                self.window_2.un_hide()
                self.window_1.hide()

                self.handle_window2()

        self.window_1.close()
        self.window_2.close()
