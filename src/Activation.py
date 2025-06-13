import numpy as np

from Layer import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.array) -> None:
        self.output = np.maximum(0, inputs)

class SoftMax(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.array) -> None:

        # input - max input by row
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
