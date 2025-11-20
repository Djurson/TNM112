import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    #TODO: specify the different activation functions
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x

    elif activation == 'relu':
        return np.maximum(0, x)

    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))

    elif activation == 'softmax':
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
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
        self.N = sum(W1.size for W1 in W) + sum(b1.size for b1 in b) 

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = np.zeros((x.shape[0], self.dataset.K))

        # TODO: implement the feed-forward layer operations
        # 1. Specify a loop over all the datapoints
        for n in range(x.shape[0]):

        # 2. Specify the input layer (2x1 matrix)
            h = x[n].reshape(2,1)

        # 3. For each hidden layer, perform the MLP operations
            for l in range(self.hidden_layers):
                
        #    - multiply weight matrix and output from previous layer
        #    - add bias vector
        #    - apply activation function
                z = self.W[l] @ h + self.b[l]
                h = activation(z, self.activation)
            
        # 4. Specify the final layer, with 'softmax' activation
            z_L = self.W[self.hidden_layers] @ h + self.b[self.hidden_layers]
            h_L = activation(z_L, 'softmax')

            y[n, :] = h_L[:,0]
        
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # TODO: formulate the training loss and accuracy of the MLP
        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class

        y_train_pred = self.feedforward(self.dataset.x_train)

        train_loss = np.mean((y_train_pred - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(y_train_pred, axis=1) == self.dataset.y_train)

        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP

        y_test_pred = self.feedforward(self.dataset.x_test)

        test_loss = np.mean((y_test_pred - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(y_test_pred, axis=1) == self.dataset.y_test)
        
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
