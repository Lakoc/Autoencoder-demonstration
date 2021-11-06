class WeightsGraph:
    def __init__(self, graph):
        self.graph = graph
        self.neurons = None
        self.lines = None
        self.weights = None
        self.draw_model()

    def draw_model(self):
        neuron11 = self.graph.DrawCircle((75, 75), 10, fill_color='black', line_color='white')
        neuron12 = self.graph.DrawCircle((75, 100), 10, fill_color='black', line_color='white')
        neuron21 = self.graph.DrawCircle((150, 75), 10, fill_color='black', line_color='white')
        line1 = self.graph.DrawLine((75, 75), (150, 75))
        line2 = self.graph.DrawLine((75, 100), (150, 75))
        text1 = self.graph.DrawText('10.5', (110, 80))
        text2 = self.graph.DrawText('8.6', (110, 95))
        self.neurons = [[neuron11, neuron12], [neuron21]]
        self.lines = [line1, line2]
        self.weights = [[[text1], [text2]]]

    def update_weight(self, layer, neuron_in, neuron_out, val):
        weight = self.weights[layer][neuron_in][neuron_out]
        self.graph.TKCanvas.itemconfig(weight, text=f'{val:.1f}')
