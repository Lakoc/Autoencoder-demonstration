import numpy as np


class WeightsGraph:
    def __init__(self, graph):
        self.graph = graph
        self.neurons = None
        self.lines = None
        self.weights = None
        self.size = (800, 400)
        self.neurons = {}
        self.weights = {}

    def init_model(self, weights):
        n_layers = len(weights) + 1
        containers_w = n_layers + 1
        max_neurons = max([layer.shape[1] for layer in weights])
        containers_h = max_neurons + 1
        layers_positions = np.linspace(0, self.size[0], containers_w)
        layers_positions = [(layers_positions[position] + layers_positions[position + 1]) / 2 for position in
                            range(layers_positions.shape[0] - 1)]
        neuron_positions = np.linspace(0, self.size[1], containers_h)
        neuron_positions = [(neuron_positions[position] + neuron_positions[position + 1]) / 2 for position in
                            range(neuron_positions.shape[0] - 1)]

        n_neurons = weights[0].shape[1]
        start_position = (max_neurons - n_neurons) // 2
        self.neurons[0] = []

        for neuron in range(n_neurons):
            position = (layers_positions[0], neuron_positions[start_position + neuron])
            neuron_graph = self.graph.DrawCircle(position, 15,
                                                 fill_color='black', line_color='white')
            self.neurons[0].append((neuron_graph, position))

        for layer in range(len(weights)):
            n_neurons = weights[layer].shape[0]
            start_position = (max_neurons - n_neurons) // 2
            self.neurons[layer + 1] = []
            self.weights[layer] = []
            for neuron in range(n_neurons):
                position = (layers_positions[layer + 1], neuron_positions[start_position + neuron])
                neuron_graph = self.graph.DrawCircle(position, 15,
                                                     fill_color='green', line_color='white')
                self.neurons[layer + 1].append((neuron_graph, position))
            for curr_index, weights_curr in enumerate(weights[layer]):
                for prev_index, weight_curr in enumerate(weights_curr):
                    weight = self.graph.DrawLine(self.neurons[layer][prev_index][1],
                                                 self.neurons[layer + 1][curr_index][1])
                    self.weights[layer].append(weight)

    def update(self, weights, biases):
        for layer in list(self.neurons.keys())[1:]:
            for n_index, neuron in enumerate(self.neurons[layer]):
                bias = biases[layer - 1][n_index][0]
                color = "#%02x%02x%02x" % (0, max(255, 255 * bias), 0)
                self.graph.TKCanvas.itemconfig(neuron[0], fill=color)
