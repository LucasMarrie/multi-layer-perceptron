import numpy as np;


"""
activation functions
"""

#tanh function
def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x);

#tanh function derivative
def tanh_prime(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x)**2;


#Rectified Linear Unit function derivative
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0);

#Rectified Linear Unit function derivative
def relu_prime(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1, 0);


#sigmoid function 
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x));

#sigmoid function derivative
def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x));


leaky_relu_alpha=0.01  #leaky relu alpha constant

# Derivative of Leaky ReLU function
def leaky_relu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, x, leaky_relu_alpha * x)

# Derivative of Leaky ReLU function
def leaky_relu_prime(x: np.ndarray) -> np.ndarray:
    dx = np.ones_like(x)
    dx[x < 0] = leaky_relu_alpha
    return dx


# Linear (Identity) function
def linear(x: np.ndarray) -> np.ndarray:
    return x

# Derivative of Linear (Identity) function
def linear_prime(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

"""
Dictionary mapping activation function names to the respective function and it's derivative
"""
activation_functions : dict[str, tuple[callable, callable]] = {
    'tanh' : (tanh, tanh_prime),
    'relu': (relu, relu_prime),
    'sigmoid' : (sigmoid, sigmoid_prime),
    'leaky_relu' : (leaky_relu, leaky_relu_prime),
    'linear' : (linear, linear_prime),
}