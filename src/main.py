from Activation import ReLU, SoftMax
from Layer import LayerDense
import nnfs
from nnfs.datasets import spiral_data
from Classifier import SoftmaxLossCategoricalCrossEntropy
import numpy as np
from Optimiser import SGD, Adam

nnfs.init()

# Create dataset
X, y = spiral_data(samples=1000, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense_1 = LayerDense(2, 512, weight_regulizer_l2=5e-4, bias_regulizer_l2=5e-4)

# Create ReLU activation (to be used with Dense layer):
activation_1 = ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense_2 = LayerDense(512, 3)

activation_2 = SoftMax()

loss_activation = SoftmaxLossCategoricalCrossEntropy()

optimiser = Adam(learning_rate=0.02, decay=5e-7)
layers = [dense_1, dense_2]

for epoch in range(10001):
    dense_1.forward(X)
    activation_1.forward(dense_1.get_output())
    dense_2.forward(activation_1.get_output())

    data_loss = loss_activation.forward(dense_2.get_output(), y)
    # TODO helper?
    regularisation_loss = (
        loss_activation.loss.l2_regularisation_loss(dense_1) +
        loss_activation.loss.l2_regularisation_loss(dense_2)
    )
    loss = data_loss + regularisation_loss

    predictions = np.argmax(loss_activation.get_output(), axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 100 == 0:
        print(f"epoch: {epoch}, " +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f}, " +
              f"data_loss: {data_loss:.3f}, " +
              f'reg_loss: {regularisation_loss:.3f}), ' +
              f"lr: {optimiser.current_learning_rate:.5f}")


    # Backward pass
    loss_activation.backward(loss_activation.get_output(), y)
    dense_2.backward(loss_activation.get_d_inputs())
    activation_1.backward(dense_2.get_d_inputs())
    dense_1.backward(activation_1.get_d_inputs())

    optimiser.handle_param_updates(layers)

# Validate the model
# Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)
# Perform a forward pass of our testing data through this layer
dense_1.forward(X_test)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation_1.forward(dense_1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense_2.forward(activation_1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense_2.output, y_test)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions == y_test)
print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

