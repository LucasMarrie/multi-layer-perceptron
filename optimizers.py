"""
File for the different Optimizers
"""

import numpy as np

"""
Abstract Optimizer Class: defines the method signature of each method
"""
class Optimizer:

    """
    Constructor
    parameters: weights: np array with the shape of the weights from the FCLayer that this class belongs to
        bias: np array with the shape of the bias from the FCLayer that this class belongs to
    """
    def __init__(self, weights: np.ndarray, bias: np.ndarray) -> None:
        raise NotImplementedError


    """
    Method for updating a layer's weights based on it current weight, the gradient of the loss function, and the learning rate
    parameters: weights: weights of the FCLayer, gradient: gradient of the loss function with respect to the weights, learning_rate: learning rate
    returns: updated weights
    """
    def update_weights(self, weights: np.ndarray, gradient: np.ndarray, learning_rate: float):
        raise NotImplementedError
    

    """
    Method for updating a layer's bias based on it current bias, the gradient of the function, and the learning rate
    parameters: bias: bias of the FCLayer, gradient: gradient of the loss function, learning_rate: learning rate
    returns: udpated bias
    """
    def update_bias(self, bias: np.ndarray, gradient: np.ndarray, learning_rate: float):
        raise NotImplementedError
    


"""
Stochastic Gradient Descent Optimizer
"""
class SGD(Optimizer):

    def __init__(self, weights, bias) -> None:
        pass


    #update weights method (see abstract class for method signature)
    def update_weights(self, weights, gradient, learning_rate):
        #step towards negative direction
        return weights - learning_rate * gradient
    

    #update bias method (see abstract class for method signature)
    def update_bias(self, bias, gradient, learning_rate):
        #step towards negative direction
        return bias - learning_rate * gradient
    


"""
Stochastic Gradient Descent with momentum Optimizer
"""
class SGDM(Optimizer):

    def __init__(self, weights, bias, momentum=0.9):
        #initialize constants for sgdm optimizer based on optimizer settings
        self.momentum = momentum                        # Momentum Coefficient

        #initialize variables for sgdm optimizer
        self.weight_velocity = np.zeros_like(weights)   # Velocity for weights
        self.bias_velocity = np.zeros_like(bias)        # Velocity for Bias


    #update weights method (see abstract class for method signature)
    def update_weights(self, weights, gradient, learning_rate):
        #update weight velocity based of current gradient and momentum
        self.weight_velocity = self.momentum * self.weight_velocity - learning_rate * gradient

        #update weights using velocity
        return weights + self.weight_velocity


    #update bias method (see abstract class for method signature)
    def update_bias(self, bias, gradient, learning_rate):
        #update bias velocity based of current gradient and momentum
        self.bias_velocity = self.momentum * self.bias_velocity - learning_rate * gradient

        #update bias using velocity
        return bias + self.bias_velocity



"""
Adaptive Gradient Optimizer
"""
class ADAGRAD(Optimizer):

    """
    Optimizer settings: 
        epsilon: ε , default value: 1e-8
    """
    def __init__(self, weights, bias, epsilon=1e-8):
        #initialize constants for adagrad optimizer based on optimizer settings
        self.epsilon = epsilon                          # Small value to avoid division by zero

        #initialize variables for adagrad optimizer
        self.weights_cache = np.zeros_like(weights)     # Cache for accumulated squared weight gradients
        self.bias_cache = np.zeros_like(bias)           # Cache for accumulated squared bias gradients


    #update weights method (see abstract class for method signature)
    def update_weights(self, weights, gradient, learning_rate):
        #accumulate squared weight gradient
        self.weights_cache += gradient ** 2

        #update weights using current gradient and adapted learning rate based off the accumulated gradients
        return weights - learning_rate * gradient / (np.sqrt(self.weights_cache) + self.epsilon)


    #update bias method (see abstract class for method signature)
    def update_bias(self, bias, gradient, learning_rate):
        #accumulate squared bias gradient
        self.bias_cache += gradient ** 2

        #update bias using current gradient and adapted learning rate based off the accumulated gradients
        return bias - learning_rate * gradient / (np.sqrt(self.bias_cache) + self.epsilon)
    


"""
Root Mean Square Propagation Optimizer
"""
class RMSPROP(Optimizer):
    
    """
    Optimizer settings: 
        beta: β , default value: 0.9
        epsilon: ε , default value: 1e-8
    """
    def __init__(self, weights, bias, beta=0.9, epsilon=1e-8):
        #initialize constants for adagrad optimizer based on optimizer settings
        self.beta = beta                                # Decay rate of accumulated gradients
        self.epsilon = epsilon                          # Small value to avoid division by zero

        #initialize variables for rmsprop optimizer
        self.weights_cache = np.zeros_like(weights)     # Cache for accumulated squared weight gradients
        self.bias_cache = np.zeros_like(bias)           # Cache for accumulated squared bias gradients


    #update weights method (see abstract class for method signature)
    def update_weights(self, weights, gradient, learning_rate):
        #update weights cache with beta and accumulate squared weight gradient
        self.weights_cache = self.beta * self.weights_cache + (1 - self.beta) * (gradient ** 2)

        #update bias using current gradient and adapted learning rate based off the accumulated gradients
        return weights - learning_rate * gradient / (np.sqrt(self.weights_cache) + self.epsilon)


    #update bias method (see abstract class for method signature)
    def update_bias(self, bias, gradient, learning_rate):
        #update bias cache with beta and accumulate squared bias gradient
        self.bias_cache = self.beta * self.bias_cache + (1 - self.beta) * (gradient ** 2)

        #update bias using current gradient and adapted learning rate based off the accumulated gradients
        return bias - learning_rate * gradient / (np.sqrt(self.bias_cache) + self.epsilon)



"""
Adaptive Moment Estimation Optimization
""" 
class ADAM(Optimizer):


    """
    Optimizer settings: 
        beta1: βm , default value: 0.9
        beta2: βv , default value: 0.999
        epsilon: ε , default value: 1e-8
    """
    def __init__(self, weights, bias, beta1=0.9, beta2=0.999, epsilon=1e-8):
        #initialize the constants for the adam optimizer based on optimizer settings
        self.beta1 = beta1                              # Decay rate for the First moment
        self.beta2 = beta2                              # Decay rate for the Second moment
        self.epsilon = epsilon                          # Small value to avoid devision by zero

        #initialize variables for adam optimizer
        self.m_weights = np.zeros_like(weights)         # First moment for weights
        self.v_weights = np.zeros_like(weights)         # Second moment for weights
        self.m_bias = np.zeros_like(bias)               # First moment for bias
        self.v_bias = np.zeros_like(bias)               # Second moment for bias
        self.timestep = 0                               # timestep


    #update weights method (see abstract class for method signature)
    def update_weights(self, weights, gradient, learning_rate):
        #increase timestep
        self.timestep += 1

        #update first moment with beta1 and accumulate weight gradient
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * gradient
        #update second moment with beta2 and accumulate squared weight gradient
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (gradient ** 2)

        #bias correction for first and second moment
        m_hat = self.m_weights / (1 - self.beta1 ** self.timestep)
        v_hat = self.v_weights / (1 - self.beta2 ** self.timestep)

        #update weights using adapted learning rate based of bias corrected first and second moments
        return weights - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


    #update bias method (see abstract class for method signature)
    def update_bias(self, bias, gradient, learning_rate):
        #update first moment with beta1 and accumulate bias gradient
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * gradient
        #update second moment with beta2 and accumulate squared bias gradient
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * (gradient ** 2)
        
        #bias correction for first and second moment
        m_hat = self.m_bias / (1 - self.beta1 ** self.timestep)
        v_hat = self.v_bias / (1 - self.beta2 ** self.timestep)

        #update bias using adapted learning rate based of bias corrected first and second moments
        return bias - learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)



"""
Dictionary mapping optimizer names to the respective optimizer class
"""
optimizers : dict[str, Optimizer] = {
    'sgd': SGD,
    'sgdm': SGDM,
    'adagrad' : ADAGRAD,
    'rmsprop' : RMSPROP,
    'adam': ADAM,
}