import numpy as np

from Exception import UnsetException
from Layer import Layer


class LossBase:
    d_inputs = None

    def forward(self, y_pred: np.array, y_true: np.array):
        raise NotImplemented()

    def calculate(self, output: np.array, y: np.array) -> np.array:
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss

    def accuracy(self, y_pred: np.array, y_true: np.array) -> float:
        predictions = np.argmax(y_pred, axis=1)

        # flatten matrix into 1-D array for 1-hot encoded predictions
        if len(y_pred.shape) == 2:
            y_true = np.argmax(y_pred, axis=1)

        return float(np.mean(predictions == y_true))

    def get_d_inputs(self):
        if self.d_inputs is None:
            raise UnsetException

        return self.d_inputs

    def l2_regularisation_loss(self, layer: Layer) -> float:
        regularisation_loss = 0

        if layer.weight_regulizer_l2 > 0:
            regularisation_loss += (
                layer.weight_regulizer_l2 * np.sum(
                layer.weights * layer.weights
                )
            )

        if layer.bias_regulizer_l2 > 0:
            regularisation_loss += (
                layer.bias_regulizer_l2 * np.sum(
                layer.biases * layer.biases
                )
            )
        return regularisation_loss


class CategoricalCrossEntropy(LossBase):
    def forward(self, y_predicted: np.array, y_true: np.array) -> np.array:
        """
        :param y_predicted: input matrix from previous dense layer
        :param y_true: ground truth vector of correct predictions

        :return negative log likelihood loss matrix
        """
        n_samples = len(y_predicted)

        # Shift data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_predicted, 1e-7, 1 - 1e-7)

        # if categorical labels
        # e.g [0, 1, 1, 2, 0, 1]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                # select true value for each index in the sample row
                range(n_samples),
                y_true
            ]
        # otherwise mask values by zeroing values that are not one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        else:
            raise Exception(f"Unhandled shape for preds: {y_true.shape}")

        # negative log of prediction at each index matching the
        # true value
        # -np.log(0.1) == 2.302
        # -np.log(0.9) == 0.105
        # rewards correct predictions with low loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


    def backward(self, d_values: np.array, y_true: np.array) -> None:
        """
        Categorical cross entropy backward step. Derivative
        is ground truth vector divided by prediction vector
        multiplied by -1

        :param d_values: derivatives passed back from next layer
        :param y_true: ground truth vector of correct predictions
        """

        n_samples = len(d_values)
        # number of labels for each sample
        n_labels = len(d_values[0])

        # if labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
                y_true = np.eye(n_labels)[y_true]

        d_inputs = -y_true / d_values
        # normalize gradient
        self.d_inputs = d_inputs / n_samples


