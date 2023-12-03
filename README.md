# Multi Layer Perceptron (MLP)

Implementation of an MLP in Python using Numpy

### How to use:
1. import mlp class  
`from mlp import MLP`

2. create mlp class instance with required arguments  
`model = MLP(4, [5,3,3], ['tanh', 'tanh', 'linear'], optimizer='sgd', loss='mse')`

3. train model  
`model.fit(X_train, y_train, epochs=500, learning_rate=0.01, batch_size=1)`

4. use model to predit  
`predicted_y = model.predict(x_test)`

### Activation Functions:
1. Linear (identity): `"linear"`
2. Relu : `"relu"`
3. Tanh : `"tanh"`
4. Leaky Relu: `"leaky_relu"`
5. Sigmoid: `"sigmoid"`

### Loss Functions:
1. Mean Squared Error: `"mse"`
2. Binary Cross Entropy: `"bce"`

### Optimizers:
1. Stochastic Gradient Descent (SGD): `"sgd"`
2. SGD with Momentum: `"sgdm"`
3. Adagrad: `"adagrad"`
4. RMSprop: `"rmsprop"`
5. Adam: `"adam"`