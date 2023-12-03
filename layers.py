import numpy as np
from activation_functions import activation_functions
from optimizers import optimizers


"""
Abstract Layer Class: defines the method signature of each method
"""
class AbstractLayer:
    def __init__(self):
        raise NotImplementedError


    """
    Forward Propogation Method
    parameters: input_data: np array of shape (batch_size, input_size)
    returns: np array of shape (batch_size, output_size)
    """
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    """
    Backward Propogation Method
    parameters: error: np array of shape (batch_size, output_size)
    """
    def backward(self, error: np.ndarray, learning_rate: float) -> np.ndarray:
        raise NotImplementedError



"""
Fully Connected Layer
"""
class FCLayer(AbstractLayer):

    """
    Constructor 
    parameters: input_size: size of input, output_size: number of nodes in layer, optimizer: name of optimizer to use, 
        optimizer settings: kwargs of optimizer settings to override (see optimizer classes)
    """
    def __init__(self, input_size: int, output_size: int, optimizer: str, **optimizer_settings: float):
        #initialize matrices for weights and bias between the inputs and the nodes in this layers
        #weights and bias are each set to a random number between -0.5 and 0.5
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

        #instantiate and set the optimizer to use based off the optimizer name, passing in the weights, bias, and optimizer settings
        self.optimizer = optimizers[optimizer](self.weights, self.bias, **optimizer_settings)


    #forward propogation method (see abstract class for method signature)
    def forward(self, input_data):
        #cache input for use in back propogation
        self.input = input_data

        #obtain output by performing dot product of input data and weights and adding bias
        output = input_data @ self.weights + self.bias
        return output
    

    #backward propogation method (see abstract class for method signature)
    def backward(self, error, learning_rate):
        #calculate error of current layer
        input_error = error @ self.weights.T

        #calculate gradients for the weights and bias
        weight_gradient = self.input.T @ error
        bias_gradient = np.sum(error, axis=0, keepdims=True)

        #input weight and bias into optimizer with respective gradients to update weights and bias
        self.weights = self.optimizer.update_weights(self.weights, weight_gradient, learning_rate)
        self.bias = self.optimizer.update_bias(self.bias, bias_gradient, learning_rate)

        #return error of current layer
        return input_error
    


"""
Activation Layer
"""
class ActivationLayer(AbstractLayer):

    """
    Constructor 
    parameters: activation_function: name of activation function to use
    """
    def __init__(self, activation_function: str):
        #set activation function and it's derivative to use based off activation function name
        self.function, self.function_prime = activation_functions[activation_function]


    #forward propogation method (see abstract class for method signature)
    def forward(self, input_data):
        #cache input for use in back propogation
        self.input = input_data

        #obtain output by passing input through activation function
        output = self.function(input_data)
        return output


    #backward propogation method (see abstract class for method signature)
    def backward(self, error, learning_rate):
        #return error for this layer by multiplying error by the derivative of the input data
        return self.function_prime(self.input) * error



