import PySimpleGUI as sg
import numpy as np


def init_layout(config):
    activation_choices = ('Tanh', 'Sigmoid', 'Linear')
    optimizer_choices = ('NoOptimizer', 'Momentum', 'RMSProp', 'Adam')

    layout1 = [[sg.Text('Data config (ensure to have corresponding dimensionality of inputs) :\n')],
               [sg.Text('Clusters (please ensure to bracket each cluster data):')],
               [sg.Text('Cluster means:')], [
                   sg.Multiline(np.array(config['data']['means']), size=(80, 5), key='means')],
               [sg.Text('Cluster variances:')],
               [sg.Multiline(
                   np.array(config['data']['variances']),
                   size=(80, 10), key='variances')],
               [sg.Text('Clusters number of points:')],
               [sg.Multiline(np.array(config['data']['counts']), size=(80, 5), key='counts')],
               [sg.Text('\nModel config:\n')],
               [sg.Text(
                   'Number of neurons in layers, make sure to create at least 3 layers.'
                   '\nLayer representing features must have exactly 2 neurons.\n'
                   'First and last layer size must equal.\nVisualisation with more than 10 neurons per layer'
                   ' or more than 9 layers might not work perfectly):')],
               [sg.Multiline(np.array(config['model']['layers']), size=(80, 5), key='layers')],
               [sg.Text('Layer index to visualize (0-N):'),
                sg.Input(config['model']['selected_layer'], key='layer')],
               [sg.Text('Activation function, used over layers:'),
                sg.Listbox(activation_choices, size=(15, len(activation_choices)), default_values=['Tanh'],
                           key='a_func')],
               [sg.Text('\nTraining config:\n')],
               [sg.Text('Batch size (1 - number of generated points):'),
                sg.Input(config['training']['batch_size'], key='batch_size')],
               [sg.Text('Max iterations:'), sg.Input(config['training']['iterations'], key='iterations')],
               [sg.Text('Epsilon:'),
                sg.Input(config['training']['epsilon'], key='epsilon')],
               [sg.Text('Optimizer:'),
                sg.Listbox(optimizer_choices, size=(15, len(optimizer_choices)), default_values=['Adam'],
                           key='optimizer')],
               [sg.Text('Beta 1:'),
                sg.Input(config['training']['beta1'], key='beta1')],
               [sg.Text('Beta 2:'),
                sg.Input(config['training']['beta2'], key='beta2')],
               [sg.Text('Learning rate:'),
                sg.Input(config['training']['eta'], key='eta')],
               [sg.Button('Done')]]
    layout2 = [
        [sg.Text("Batch: 0\nEpoch: 0", key='Counter', expand_x=True), sg.Button("Edit config")],
        [sg.Canvas(key="CANVAS_FEATURES", expand_x=True, size=(400, 400)),
         sg.Canvas(key="CANVAS_LOSS", size=(400, 400))],
        [sg.Graph(canvas_size=(800, 400), graph_bottom_left=(0, 0), graph_top_right=(800, 400),
                  background_color='white',
                  key='GRAPH_WEIGHTS'), sg.Canvas(key="CANVAS_CMAP", size=(100, 400))],
        [sg.Button("Single Batch"), sg.Button("Epoch"), sg.Button("100 epochs"), sg.Button("Train")],
    ]
    return layout1, layout2
