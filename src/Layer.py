import numpy as np

from Exception import UnsetException

class Layer:
    def __init__(self):
        self.output = None

        # batch of inputs with m inputs x n features
        self.inputs = None
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None
        self.d_inputs = None
        self.weight_momentums = None
        self.bias_momentums = None
        self.weight_cache = None
        self.bias_cache = None

    def get_output(self) -> np.array:
        if self.output is None:
            raise UnsetException()
        return self.output

    def get_inputs(self) -> np.array:
        if self.inputs is None:
            raise UnsetException
        return self.inputs

    def get_weights(self):
        if self.weights is None:
            raise UnsetException
        return self.weights

    def get_biases(self):
        if self.biases is None:
            raise UnsetException
        return self.biases

    def get_d_weights(self) -> np.array:
        if self.d_weights is None:
            raise UnsetException
        return self.d_weights

    def get_d_biases(self) -> np.array:
        if self.d_biases is None:
            raise UnsetException
        return self.d_biases

    def get_d_inputs(self) -> np.array:
        if self.d_inputs is None:
            raise UnsetException
        return self.d_inputs

    def get_weight_momentums(self) -> np.array:
        if self.weight_momentums is None:
            self.weight_momentums = np.zeros_like(self.get_weights())
        return self.weight_momentums

    def get_bias_momentums(self) -> np.array:
        if self.bias_momentums is None:
            self.bias_momentums = np.zeros_like(self.get_biases())
        return self.bias_momentums

    def get_weight_cache(self) -> np.array:
        if self.weight_cache is None:
            self.weight_cache = np.zeros_like(self.get_weights())
        return self.weight_cache

    def get_bias_cache(self) -> np.array:
        if self.bias_cache is None:
            self.bias_cache = np.zeros_like(self.get_biases())
        return self.bias_cache

    def set_weights(self, new_weights: np.array) -> None:
        self.weights = new_weights

    def set_biases(self, new_biases: np.array) -> None:
        self.biases = new_biases

    def set_weight_momentums(self, new_weight_momentums: np.array) -> None:
        self.weight_momentums = new_weight_momentums

    def set_bias_momentums(self, new_bias_momentums: np.array) -> None:
        self.bias_momentums = new_bias_momentums

    def set_weight_cache(self, new_weight_cache: np.array) -> None:
        self.weight_cache = new_weight_cache

    def set_bias_cache(self, new_bias_cache: np.array) -> None:
        self.bias_cache = new_bias_cache

class LayerDense(Layer):
    def __init__(self, n_inputs: int, n_neurons: int):
        super().__init__()
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases  = np.zeros((1, n_neurons))
        self.output = None

    def forward(self, inputs: np.array) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_values: np.array) -> np.array:
        """
        :param d_values: the derivatives passed back from the next layer
        :return:
        """

        # derivative is the other side of weight x inputs, multiplied
        # with next function's derivative (by the chain rule)
        self.d_weights = np.dot(self.get_inputs().T, d_values)
        self.d_inputs = np.dot(d_values, self.get_weights().T)

        # derivative of summed bias is just 1, so no need to multiply,
        # just sum incoming derivatives from next layer column-wise
        # note, result matches shape of biases vector, or 1 x n_features
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
