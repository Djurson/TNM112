import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
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
    def feedforward( self: 'MLP', x):
        y = np.zeros((len(x), self.dataset.K))

        for n in range(len(x)):
            h = x[n].reshape(2, 1)

            for l in range(self.hidden_layers):
                z = self.W[l] @ h + self.b[l] # Lecture 3: Slide 31
                h = activation(z, self.activation)
            
            z_out = self.W[-1] @ h + self.b[-1]
            y[n, :] = activation(z_out, 'softmax').flatten()
        
        return y

    # Measure performance of model
    def evaluate(self: 'MLP'):
        print('Model performance:')

        output_train = self.feedforward(self.dataset.x_train)        

        train_loss = np.mean((output_train - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(output_train, axis=1) == self.dataset.y_train) * 100

        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        output_test = self.feedforward(self.dataset.x_test)

        test_loss = np.mean((output_test - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(output_test, axis=1) == self.dataset.y_test) * 100
        
        print("\tTest loss:      %0.4f"%test_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
