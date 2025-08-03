from typing import List

import numpy as np

from Layer import Layer

class Optimiser:
    learning_rate = 1.0

    def __init__(self, learning_rate = 1.0, decay = 1.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def handle_param_updates(self, layers: List[np.array]):
        self.pre_update_parmas()
        for layer in layers:
            self.update_params(layer)
        self.post_update_params()

    def pre_update_parmas(self) -> None:
        """
        reduce learning rate as number iterations increase
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                    1.0 / (1.0 + self.decay * self.iterations)
            )

    def update_params(self, layer: Layer):
        raise NotImplementedError

    def post_update_params(self):
        self.iterations += 1

class SGD(Optimiser):
    learning_rate = 1.0

    def __init__(self, learning_rate=1.0, decay=0.0, momentum_factor=0.0):
        super().__init__(learning_rate, decay)
        self.momentum_factor = momentum_factor

    def update_params(self, layer: Layer) -> None:
        old_weights = layer.get_weights()
        old_biases = layer.get_biases()

        if self.momentum_factor:
            old_weight_momentums = layer.get_weight_momentums()
            old_bias_momentums = layer.get_bias_momentums()

            # take previous updates multiplied by momentum retention factor
            # and update with new gradients
            weight_updates = (self.momentum_factor * old_weight_momentums) - (
                    self.current_learning_rate * layer.get_d_weights())
            layer.set_weight_momentums(weight_updates)

            bias_updates = (self.momentum_factor * old_bias_momentums) - (
                    self.current_learning_rate * layer.get_d_biases())
            layer.set_bias_momentums(bias_updates)

        else:
            weight_updates = -self.current_learning_rate * layer.get_d_weights()
            bias_updates = -self.current_learning_rate * layer.get_d_biases()

        layer.set_weights(old_weights + weight_updates)
        layer.set_biases(old_biases + bias_updates)

class Adam(Optimiser):
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        super().__init__(learning_rate, decay)
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2


    def update_params(self, layer: Layer):
        old_weight_momentums = layer.get_weight_momentums()
        old_bias_momentums = layer.get_bias_momentums()
        old_weight_cache = layer.get_weight_cache()
        old_bias_cache = layer.get_bias_cache()

        # update momentum with boosted gradients
        new_weight_momentums = self.beta_1 * old_weight_momentums + (1 - self.beta_1) * layer.get_d_weights()
        new_bias_momentums = self.beta_1 * old_bias_momentums + (1 - self.beta_1) * layer.get_d_biases()

        layer.set_weight_momentums(new_weight_momentums)
        layer.set_bias_momentums(new_bias_momentums)

        # correct momentum
        weight_momentums_corrected = layer.get_weight_momentums() / (
                1 - self.beta_1 ** (self.iterations + 1)
        )
        bias_momentums_corrected = layer.get_bias_momentums() / (
                1 - self.beta_1 ** (self.iterations + 1)
        )

        # update cache with squared current gradients
        new_weight_cache = self.beta_2 * old_weight_cache + (
                (1 - self.beta_2) * layer.get_d_weights() ** 2
        )
        new_bias_cache = self.beta_2 * old_bias_cache + (
                           (1 - self.beta_2) * layer.get_d_biases() ** 2
        )
        layer.set_weight_cache(new_weight_cache)
        layer.set_bias_cache(new_bias_cache)

        # corrected cache
        weight_cache_corrected = layer.weight_cache / (
                1 - self.beta_2 ** (self.iterations + 1)
        )
        bias_cache_corrected = layer.bias_cache /  (
                1 - self.beta_2 ** (self.iterations + 1)
        )

        # Normal SGD update + normalisation with square root cache
        weight_updates = -self.current_learning_rate * (
                         weight_momentums_corrected /
                         (np.sqrt(weight_cache_corrected) +
                          self.epsilon)
        )
        bias_updates = -self.current_learning_rate * (
                        bias_momentums_corrected /
                        (np.sqrt(bias_cache_corrected) +
                         self.epsilon)
        )
        layer.set_weights(layer.get_weights() + weight_updates)
        layer.set_biases(layer.get_biases() + bias_updates)
        # print(f"NEW WEIGHTS: {np.mean(layer.get_weights())}")















