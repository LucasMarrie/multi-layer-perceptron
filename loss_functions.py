"""
File For Loss Functions
"""


import numpy as np

"""
Loss Functions
"""

#Mean Squared Error Loss Function
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.power(y_true-y_pred, 2))

#Mean Squared Error Loss Function Derivative
def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return 2*(y_pred-y_true)/y_true.size


epsilon = 1e-15 # Small value to avoid division by zero, used for bce loss functions

# Binary cross-entropy loss function
def bce(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted values to avoid log(0)
    return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Derivative of binary cross-entropy loss function with respect to predictions
def bce_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predicted values to avoid log(0)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))


"""
Dictionary mapping loss function name to the respective function and it's derivative
"""
loss_functions : dict[str, tuple[callable, callable]] = {
    'mse': (mse, mse_prime),
    'bce': (bce, bce_prime),
}
