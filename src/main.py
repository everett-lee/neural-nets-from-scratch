from Activation import ReLU, SoftMax
from Layer import LayerDense
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense(2, 3)

# Create ReLU activation (to be used with Dense layer):
activation1 = ReLU()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = LayerDense(3, 3)


activation2 = SoftMax()

# Make a forward pass of our training data through this layer
dense1.forward(X)

# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.get_output())

dense2.forward(activation1.get_output())

activation2.forward(dense2.get_output())

print(dense2.get_output()[:5])
print(activation2.get_output()[:5])