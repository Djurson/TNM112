import numpy as np
import h5py
import matplotlib.pyplot as plt

#-------------------------------
# Data generator class
#-------------------------------
class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def generate(self, dataset='patchcam', N_train=None, N_valid=0.1):
        self.N_train = N_train
        self.N_valid = N_valid
        self.dataset = dataset

        if dataset == 'patchcam':
            # Load from h5 files (assumes they exist in a folder named 'patchcam')
            try:
                with h5py.File('Lab2/patchcam/train.h5','r') as f:
                    self.x_train = f['x'][:]
                    self.y_train = f['y'][:]
                with h5py.File('Lab2/patchcam/valid.h5','r') as f:
                    self.x_valid = f['x'][:]
                    self.y_valid = f['y'][:]
                with h5py.File('Lab2/patchcam/test_x.h5','r') as f:
                    self.x_test = f['x'][:]
                    self.y_test = [] # No labels for test set
            except OSError:
                print("Error: Could not load PatchCam files. Make sure the 'patchcam' folder exists.")
                # Fallback for testing if files are missing (Random Noise)
                print("Generating dummy data for testing code...")
                self.x_train = np.random.rand(100, 96, 96, 3).astype('float32')
                self.y_train = np.random.randint(0, 2, (100, 1))
                self.x_valid = np.random.rand(20, 96, 96, 3).astype('float32')
                self.y_valid = np.random.randint(0, 2, (20, 1))
                self.x_test = np.random.rand(20, 96, 96, 3).astype('float32')
                self.y_test = []

            self.normalize()
        else:
            raise Exception("This PyTorch implementation currently supports 'patchcam' dataset.")

        # Number of classes
        self.K = len(np.unique(self.y_train))
        
        # Number of color channels
        self.C = self.x_train.shape[3]
        
        # One hot encoding of class labels (Numpy implementation)
        self.y_train_oh = self._to_categorical(self.y_train, self.K)
        self.y_valid_oh = self._to_categorical(self.y_valid, self.K)
        # Test set has no labels to encode
        '''
        print ("test")
        print(self.y_train_oh)
        print(self.y_valid_oh)
        '''

        if self.verbose:
            print('Data specification:')
            print('\tDataset type:          ', self.dataset)
            print('\tNumber of classes:     ', self.K)
            print('\tNumber of channels:    ', self.C)
            print('\tTraining data shape:   ', self.x_train.shape)
            print('\tValidation data shape: ', self.x_valid.shape)
            print('\tTest data shape:       ', self.x_test.shape)

    def split_data(self):
        # (Used if loading a single big dataset, e.g. MNIST)
        N = self.x_train.shape[0]
        ind = np.random.permutation(N)
        self.x_train = self.x_train[ind]
        self.y_train = self.y_train[ind]

        self.N_valid = int(N*self.N_valid)
        N = N - self.N_valid
        self.x_valid = self.x_train[-self.N_valid:]
        self.y_valid = self.y_train[-self.N_valid:]

        if self.N_train and self.N_train < N:
            self.x_train = self.x_train[:self.N_train]
            self.y_train = self.y_train[:self.N_train]
        else:
            self.x_train = self.x_train[:N]
            self.y_train = self.y_train[:N]
            self.N_train = N

    def normalize(self):
        # Normalize to [-1, 1]
        self.x_train = 2*self.x_train.astype("float32") / 255 - 1.0
        self.x_valid = 2*self.x_valid.astype("float32") / 255 - 1.0
        self.x_test = 2*self.x_test.astype("float32") / 255 - 1.0

    def plot(self, xx=12, yy=3, save_path=None):
        plt.figure(figsize=(18, yy*2))
        cm = 'gray' if self.C==1 else 'viridis'
        for i in range(xx*yy):
            plt.subplot(yy, xx, i+1)
            # Rescale back to [0,1] for plotting
            img = (self.x_train[i] + 1) / 2
            plt.imshow(img, cmap=cm)
            plt.title('label=%d'%(self.y_train[i]))
            plt.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _to_categorical(self, y, num_classes):
        """ Numpy implementation of keras.utils.to_categorical """
        return np.eye(num_classes)[y.reshape(-1)]
    
    # test
    '''
    def _to_categorical(self, y, num_classes):
        """ Numpy implementation of keras.utils.to_categorical """
        return np.eye(num_classes)[y.reshape(-1)]
    '''