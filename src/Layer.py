import numpy as np

from Exception import OutputUnsetException

class Layer:
    def __init__(self):
        self.output = None

    def get_output(self) -> np.array:
        if self.output is None:
            raise OutputUnsetException()
        return self.output

class LayerDense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs: np.array) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

