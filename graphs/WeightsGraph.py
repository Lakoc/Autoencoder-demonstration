import numpy as np
import matplotlib


class WeightsGraph:
    """Maintains Neural network graphical representation on the canvas."""

    def __init__(self, graph):
        self.graph = graph
        self.neurons = None
        self.lines = None
        self.weights = None
        self.size = (800, 400)
        self.neurons = {}
        self.weights = {}
        self.c_map = matplotlib.cm.get_cmap('RdYlBu')
        self.normalizer = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        self.layers_positions = None
        self.neuron_positions = None
        self.max_neurons = 0

    def draw_arrow(self, start_point, end_point, width):
        """Auxiliary function that draws arrow"""
        arr_len = end_point[0] - start_point[0]
        wing_len = 1 / 4 * arr_len
        self.graph.DrawLine(start_point, end_point, width=width)
        self.graph.DrawLine(end_point, end_point - wing_len, width=width)
        self.graph.DrawLine(end_point, [end_point[0] - wing_len, end_point[1] + wing_len], width=width)

    def calculate_positions(self, weights):
        n_layers = len(weights) + 1
        containers_w = n_layers + 1
        self.max_neurons = max([layer.shape[1] for layer in weights])
        containers_h = self.max_neurons + 1
        self.layers_positions = np.linspace(0, self.size[0], containers_w)
        self.layers_positions = [(self.layers_positions[position] + self.layers_positions[position + 1]) / 2 for
                                 position in
                                 range(self.layers_positions.shape[0] - 1)]
        self.neuron_positions = np.linspace(0, self.size[1], containers_h)
        self.neuron_positions = [(self.neuron_positions[position] + self.neuron_positions[position + 1]) / 2 for
                                 position in
                                 range(self.neuron_positions.shape[0] - 1)]

    def init_first_layer(self, start_position, number_of_neurons):
        for neuron in range(number_of_neurons):
            position = (self.layers_positions[0], self.neuron_positions[start_position + neuron])
            self.draw_arrow([position[0] - 45, position[1]], [position[0] - 20, position[1]], 1)
            self.neurons[0].append(position)

    def process_other_layers(self, weights):
        for layer in range(len(weights)):
            n_neurons = weights[layer].shape[0]
            start_position = (self.max_neurons - n_neurons) // 2
            self.neurons[layer + 1] = []
            self.weights[layer] = {}
            for neuron in range(n_neurons):
                position = (self.layers_positions[layer + 1], self.neuron_positions[start_position + neuron])
                self.neurons[layer + 1].append(position)
                if layer == len(weights) - 1:
                    self.draw_arrow([position[0] + 20, position[1]], [position[0] + 45, position[1]], 1)

            for curr_index, weights_curr in enumerate(weights[layer]):
                self.weights[layer][curr_index] = {}
                for prev_index, weight_curr in enumerate(weights_curr):
                    weight = self.graph.DrawLine(self.neurons[layer][prev_index],
                                                 self.neurons[layer + 1][curr_index], width=5)
                    self.weights[layer][curr_index][prev_index] = weight

    def draw_neurons(self):
        for key, layer in self.neurons.items():
            for l_p, position in enumerate(layer):
                neuron = self.graph.DrawCircle(position, 15,
                                               fill_color='white', line_color='black')
                self.neurons[key][l_p] = (neuron, position)

    def init_model(self, weights):
        """Init neural network model on specified canvas based on provided weights"""
        self.neurons[0] = []
        n_neurons = weights[0].shape[1]
        start_position = (self.max_neurons - n_neurons) // 2
        self.calculate_positions(weights)
        self.init_first_layer(start_position, n_neurons)
        self.process_other_layers(weights)
        self.draw_neurons()

    def update(self, weights, biases):
        """Update colors and size of objects on canvas based on current model state"""
        for layer_index, layer in enumerate(weights):
            for curr_index, neuron in enumerate(layer.T):
                for next_index, value in enumerate(neuron):
                    # Conversion to Canvas rgb representation
                    rgb = "#%02x%02x%02x" % tuple([int(val * 255) for val in self.c_map(self.normalizer(value))[:-1]])
                    weight = np.abs(value) * 5 + 1
                    self.graph.TKCanvas.itemconfig(self.weights[layer_index][next_index][curr_index], fill=rgb,
                                                   width=weight)

        for layer in list(self.neurons.keys())[1:]:
            for n_index, neuron in enumerate(self.neurons[layer]):
                bias = biases[layer - 1][n_index]
                rgb = "#%02x%02x%02x" % tuple([int(val * 255) for val in self.c_map(self.normalizer(bias))[:-1]])
                self.graph.TKCanvas.itemconfig(neuron[0], fill=rgb)
