import numpy as np

class LossBase:

    def forward(self, y_pred: np.array, y_true: np.array):
        raise NotImplemented()

    def calculate(self, output: np.array, y: np.array) -> np.array:
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss


class CategoricalCrossEntropy(LossBase):
    def forward(self, y_pred: np.array, y_true: np.array):

        n_samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values -
        # only if categorical labels
        # e.g [0, 1, 1, 2, 0, 1]
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                # For each index in the sample row
                range(n_samples),
                y_true
            ]
        # Mask values by zeroing values that are not one-hot encoded
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )
        else:
            raise Exception(f"Unhandled shape for preds: {y_true.shape}")

        # negative log of prediction at each index matching the provided
        # true value
        # -np.log(0.1) == 2.302
        # -np.log(0.9) == 0.105
        # rewards correct reductions with low loss
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods