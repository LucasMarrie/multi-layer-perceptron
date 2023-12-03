"""
File for the MLP (Multi Layer Perceptron Class)
"""

import numpy as np
from layers import FCLayer, ActivationLayer
from loss_functions import loss_functions


"""
Multi Layer Perceptron Class
"""
class MLP:
    """
    Constructor
    parameters: input_size: size of input data, layer_nodes: list of nodes in each layer, activation functions: list of activation function after each layer,
        optimizer: name of the optimizer to use, loss function: name of the loss function to use, optimizer_settings: kwargs of optimizer settings to override (see optimizer classes)
    """
    def __init__(self, input_size: int, layer_nodes: list[int], activation_functions: list[str], optimizer: str = 'sgd', loss_function: str = 'mse', **optimizer_settings: float) -> None:
        #create and set the layers and set the loss function
        self.set_layers(input_size, layer_nodes, activation_functions, optimizer, **optimizer_settings)
        self.set_loss(loss_function)


    #create and set the layers of the mlp
    def set_layers(self, input_size: int, layer_nodes: list[int],  activation_functions: list[str], optimizer: str, **optimizer_settings: float) -> None:
        #define input and output size for the class
        self.input_size = input_size
        self.output_size = layer_nodes[-1]
        #define new list of layers 
        self.layers = []

        #Add Layers with the input size set to the output of the previous layer
        layer_input = input_size        
        for i in range(len(layer_nodes)):
            layer_output = layer_nodes[i]
            #Add Fully Connected layer with predefined input and output sizes, optimizer name and optimizer settings
            self.layers.append(FCLayer(layer_input, layer_output, optimizer, **optimizer_settings))
            #Add Activation Layer with predefined activation function
            self.layers.append(ActivationLayer(activation_functions[i]))
            layer_input = layer_output


    #set the loss function to use in training
    def set_loss(self, loss_function: str) -> None:
        #set loss function and it's derivative to use based off loss function name
        loss, loss_prime = loss_functions[loss_function]
        self.loss = loss
        self.loss_prime = loss_prime


    """
    Method for running a set of input data through the model and obtaining predictions
    parameters: input_data: 2 dimensional np array (row: element, col: features)
    returns: 2 dimensional np array (row: element, col: predicted outputs)
    """#
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        #Instantiate matrix to store predictions
        predictions = np.zeros((len(input_data), self.output_size))
        #loop through each input
        for i in range(len(input_data)):
            #select input while mainting desired matrix shape (1, f) where f is the amount of features
            x = input_data[[i], :]
            #run input through model
            predictions[i] = self.forward(x)

        return predictions


    """
    Method for training the model on a set of input data and the target labels
    parameters: input_data: 2 dimensional np array (row: element, col: features), targets: 2 dimensional np array (row: element, col: target outputs),
        epochs: number of epochs, learning_rate: optimizer learning rate, batch_size: size of each mini-batch
        print_progress: boolean to control whether training progress is printed to stdout
    """
    def fit(self, input_data: np.ndarray, targets: np.ndarray, epochs: int, learning_rate: float, batch_size: int = 1, print_progress : bool = True) -> None:
        #cache frequently used methods, samples and batch count
        samples = len(input_data)
        batch_count = np.ceil(samples / batch_size)

        #tracking loss data over training
        self.loss_list = []
        loss_samples = 100

        #run training loop for number of epochs
        for epoch in range(epochs):
            indices = np.random.permutation(samples)                #randomise the training data by shuffling the indices
            batches = np.array_split(indices, batch_count)          #seperate the indices into batches based off the batch size
            loss = 0                                                #reset loss for tracking training progress

            #loop through all batches
            for batch in batches:
                output = self.forward(input_data[batch, :])         #select data using batch indices and do forward pass through model
                loss += self.loss(targets[batch], output)           #accumulate loss for tracking training progress
                error = self.loss_prime(targets[batch], output)     #calculate loss derivative for backpropogation
                self.backward(error, learning_rate)                 #perform backpropogation on batch

            #every 10% of epochs print training progress (epoch and mean loss of given epoch)
            if print_progress and (epoch + 1) % (epochs//10) == 0:
                print(f'[{epoch + 1}/{epochs}] Epochs, Error: {loss/batch_count}')

            #track loss over the course of epoch training
            if (epoch) % (epochs//loss_samples) == 0 or epoch == epochs-1:
                self.loss_list.append((epoch+1, np.mean(loss/batch_count)))


    #forward propogation of input through each layer
    def forward(self, x: np.ndarray) -> np.ndarray:
        #loop through each layer and call it's forward propogation function
        for layer in self.layers:
            #feed the output to the input of the next layer
            x = layer.forward(x)
        return x
    

    #backward propogation of error through each layer
    def backward(self, error: np.ndarray, learning_rate: int) -> None:
        #loop through each layer in reverse and call it's back propogation function
        for layer in reversed(self.layers): 
            #feed the output to the input of the next layer
            error = layer.backward(error, learning_rate)


            