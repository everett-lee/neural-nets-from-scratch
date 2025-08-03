from Activation import ReLU, SoftMax
from Layer import LayerDense
import nnfs
from nnfs.datasets import spiral_data
from Classifier import SoftmaxLossCategoricalCrossEntropy
import numpy as np
from Optimiser import SGD, Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense_1 = LayerDense(2, 64)

# Create ReLU activation (to be used with Dense layer):
activation_1 = ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense_2 = LayerDense(64, 3)

activation_2 = SoftMax()

loss_activation = SoftmaxLossCategoricalCrossEntropy()

optimiser = Adam(learning_rate=0.02, decay=1e-5)
layers = [dense_1, dense_2]

for epoch in range(10001):
    dense_1.forward(X)
    activation_1.forward(dense_1.get_output())
    dense_2.forward(activation_1.get_output())
    loss = loss_activation.forward(dense_2.get_output(), y)

    predictions = np.argmax(loss_activation.get_output(), axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, " +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"lr: {optimiser.current_learning_rate}")


    # Backward pass
    loss_activation.backward(loss_activation.get_output(), y)
    dense_2.backward(loss_activation.get_d_inputs())
    activation_1.backward(dense_2.get_d_inputs())
    dense_1.backward(activation_1.get_d_inputs())

    # print("***")
    # print("Gradients:")
    # print("dense_1 d_weights: ", dense_1.get_d_weights())
    # print("dense_1 d_biases: ",dense_1.get_d_biases())
    # print("dense_2 d_weights: ",dense_2.get_d_weights())
    # print("dense_2 d_biases: ",dense_2.get_d_biases())

    optimiser.handle_param_updates(layers)

