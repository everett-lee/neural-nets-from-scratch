from Activation import SoftMax
from Exception import UnsetException
from Loss import CategoricalCrossEntropy
import numpy as np


class SoftmaxLossCategoricalCrossEntropy:
    """
    Softmax classifier - combines Softmax activation
    and cross-entropy loss for faster backward step
    """

    def __init__(self):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy()
        self.output = None
        self.d_inputs = None

    def get_d_inputs(self) -> np.array:
        if self.d_inputs is None:
            raise UnsetException

        return self.d_inputs

    def get_output(self):
        if self.output is None:
            raise UnsetException

        return self.output

    def forward(self, inputs: np.array, y_true: np.array) -> np.array:
        """
        Perform combined forward steps

        :param inputs: input matrix from previous dense layer
        :param y_true: ground truth vector of correct predictions
        :return: matrix of n samples x m loss values per sample
        """
        # output layer activation function
        self.activation.forward(inputs)

        self.output = self.activation.get_output()

        # use output predictions to calculate loss
        return self.loss.calculate(self.output, y_true)


    def backward(self, d_values: np.array, y_true: np.array) -> None:
        """
        Calculate derivative of Softmax and categorical cross entropy loss

        :param d_values: calculated loss
        :param y_true: ground truth vector of correct predictions
        :return: derivative of loss values, with shape of loss matrix
        """
        # number of training samples in batch
        n_samples = len(d_values)

        # if labels are one-hot encoded in matrix, then flatten to categorial
        if len(y_true.shape) == 2:
            # get index of that hot value, 1
            y_true = np.argmax(y_true, axis=1)

        # copy so we can mutate safely
        self.d_inputs = d_values.copy()

        # calculate gradient
        # the derivative is (predicted - ground truth)
        # when predicting categories, ground truth is just 1 at the expected index
        # so we can subtract 1 at position of the correct prediction
        self.d_inputs[range(n_samples), y_true] -= 1

        # normalise gradient
        self.d_inputs = self.d_inputs / n_samples