from typing import List

import numpy as np

from Layer import Layer


class SGD:
    learning_rate = 1.0

    def __init__(self, learning_rate: 1.0, decay = 0.0, momentum_factor = 0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum_factor = momentum_factor

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
            new_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))
            self.current_learning_rate = new_rate

    def post_update_params(self):
        self.iterations += 1

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

