import numpy as np
import matplotlib


class WeightsGraph:
    def __init__(self, graph):
        self.graph = graph
        self.neurons = None
        self.lines = None
        self.weights = None
        self.size = (800, 400)
        self.neurons = {}
        self.weights = {}
        self.cmap = matplotlib.cm.get_cmap('RdYlBu')
        self.normalizer = matplotlib.colors.Normalize(vmin=-1, vmax=1)

    def draw_arrow(self, start_point, end_point, width):
        arr_len = end_point[0] - start_point[0]
        wing_len = 1 / 4 * arr_len
        self.graph.DrawLine(start_point, end_point, width=width)
        self.graph.DrawLine(end_point, end_point - wing_len, width=width)
        self.graph.DrawLine(end_point, [end_point[0] - wing_len, end_point[1] + wing_len], width=width)

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
                                                 fill_color='white', line_color='black')
            self.draw_arrow([position[0] - 45, position[1]], [position[0] - 20, position[1]], 1)
            self.neurons[0].append((neuron_graph, position))

        for layer in range(len(weights)):
            n_neurons = weights[layer].shape[0]
            start_position = (max_neurons - n_neurons) // 2
            self.neurons[layer + 1] = []
            self.weights[layer] = {}
            for neuron in range(n_neurons):
                position = (layers_positions[layer + 1], neuron_positions[start_position + neuron])
                neuron_graph = self.graph.DrawCircle(position, 15,
                                                     fill_color='white', line_color='black')
                self.neurons[layer + 1].append((neuron_graph, position))
                if layer == len(weights) - 1:
                    self.draw_arrow([position[0] + 20, position[1]], [position[0] + 45, position[1]], 1)

            for curr_index, weights_curr in enumerate(weights[layer]):
                self.weights[layer][curr_index] = {}
                for prev_index, weight_curr in enumerate(weights_curr):
                    weight = self.graph.DrawLine(self.neurons[layer][prev_index][1],
                                                 self.neurons[layer + 1][curr_index][1], width=5)
                    self.weights[layer][curr_index][prev_index] = weight

    def update(self, weights, biases):
        for layer in list(self.neurons.keys())[1:]:
            for n_index, neuron in enumerate(self.neurons[layer]):
                bias = biases[layer - 1][n_index][0]
                rgb = "#%02x%02x%02x" % tuple([int(val * 255) for val in self.cmap(bias)[:-1]])
                self.graph.TKCanvas.itemconfig(neuron[0], fill=rgb)

        for layer_index, layer in enumerate(weights):
            for curr_index, neuron in enumerate(layer.T):
                for next_index, value in enumerate(neuron):
                    rgb = "#%02x%02x%02x" % tuple([int(val * 255) for val in self.cmap(self.normalizer(value))[:-1]])
                    self.graph.TKCanvas.itemconfig(self.weights[layer_index][next_index][curr_index], fill=rgb)
