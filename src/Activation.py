import numpy as np

from Exception import UnsetException


class Activation:
    def __init__(self):
        self.output = None

        # batch of outputs from activation layer
        self.inputs = None

        self.d_inputs = None

    def get_output(self) -> np.array:
        if self.output is None:
            raise UnsetException
        return self.output

    def get_inputs(self) -> np.array:
        if self.inputs is None:
            raise UnsetException
        return self.inputs

    def get_d_inputs(self) -> np.array:
        if self.d_inputs is None:
            raise UnsetException
        return self.d_inputs


class ReLU(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.array) -> None:
        self.inputs = inputs
        # ReLU simply clips input to range [0, inf)
        self.output = np.maximum(0, inputs)

    def backward(self, d_values: np.array) -> None:
        # mutating d_values, so create copy
        self.d_inputs = d_values.copy()

        # derivative of max is 1 here if val is > 0, else 0 (p.187)
        # these are then multiplied by derivative passed
        # back from next layer
        # so just zero out copied values where input is <= 0
        self.d_inputs[self.inputs <= 0] = 0

class SoftMax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: np.array) -> None:
        # input minus max input by row to scale
        # this prevents 'exploding' values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # exponentiate and normalise, so have range [0-1] and sum 1
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


    def backward(self, d_values: np.array) -> None:
        """
        Calculate partial derivatives of softmax. P. 226

        :param d_values: derivatives passed back from next layer
        """

        # values will be overwritten so don't matter
        self.d_inputs = np.empty_like(d_values)

        # enumerate outputs and gradients.
        for index, (single_output, single_d_values) in \
                enumerate(zip(self.output, d_values)):
            # iterate sample-wise over output and passed back gradients
            # to calculate partial derivatives
            # single_output: row of output matrix
            # single_d_values: row of d_values matrix

            # flatten single_output to shape (n_cols, 1)
            single_output = single_output.reshape(-1, 1)
            # Jacobian matrix of the output
            # note np.diagflat is equivalent to identity matrix * single_output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)

            # calculate sample-wise gradient
            # and add it to the array of sample gradients

            # multiply each row in jacobian matrix with
            # value from gradient array, to yield 2D array
            self.d_inputs[index] = np.dot(jacobian_matrix,
                                         single_d_values)