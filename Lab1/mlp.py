import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x

    elif activation == 'relu':
        return np.maximum(0, x)

    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))

    elif activation == 'softmax':
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self: 'MLP',
        dataset: data_generator.DataGenerator,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self: 'MLP',
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation

        # TODO: specify the number of hidden layers based on the length of the provided lists
        self.hidden_layers = len(W) - 1

        self.W = W # -> Lecture 2 slide 21
        self.b = b

        # TODO: specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum(Wl.size for Wl in W) + sum(bl.size for bl in b) 

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self: 'MLP',
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = np.zeros((len(x), self.dataset.K))

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        for n, xn in enumerate(x):

        # 2. Specify the input layer (2x1 matrix)
            h = xn.reshape(-1, 1)

        # 3. For each hidden layer, perform the MLP operations
            for l in range(self.hidden_layers):

        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function

        #   Lecture 3: Slide 31
                z = self.W[l] @ h + self.b[l]
                h = activation(z, self.activation)
            
        # 4. Specify the final layer, with 'softmax' activation
            z_out = self.W[-1] @ h + self.b[-1]
            y[n, :] = activation(z_out, 'softmax').flatten()
        
        return y

    # Measure performance of model
    def evaluate(self: 'MLP'):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class

        output_train = self.feedforward(self.dataset.x_train)        
        output_test = self.feedforward(self.dataset.x_test)

        train_loss = np.mean((output_train - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(output_test, axis=1) == self.dataset.y_train) * 100

        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP

        test_loss = np.mean((output_test - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(output_test, axis=1) == self.dataset.y_test) * 100
        
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
