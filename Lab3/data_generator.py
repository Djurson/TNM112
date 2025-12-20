import numpy as np
import h5py
from tensorflow import keras
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def generate(
        self,
        dataset='mnist',
        N_train=None,
        N_valid=0.1
    ):
        self.N_train = N_train
        self.N_valid = N_valid
        self.dataset = dataset

        if dataset == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
            self.split_data()
            self.normalize()

            self.x_train = np.expand_dims(self.x_train, -1)
            self.x_valid = np.expand_dims(self.x_valid, -1)
            self.x_test = np.expand_dims(self.x_test, -1)

        elif dataset == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
            self.split_data()
            self.normalize()

        elif dataset == 'patchcam':
            with h5py.File('patchcam/train.h5','r') as f:
                self.x_train = f['x'][:]
                self.y_train = f['y'][:]
            with h5py.File('patchcam/valid.h5','r') as f:
                self.x_valid = f['x'][:]
                self.y_valid = f['y'][:]
            with h5py.File('patchcam/test_x.h5','r') as f:
                self.x_test = f['x'][:]
                self.y_test = []

            self.normalize()
        else:
            raise Exception("Unknown dataset", dataset)

        self.K = len(np.unique(self.y_train))
       
        self.C = self.x_train.shape[3]
       
        self.y_train_oh = keras.utils.to_categorical(self.y_train, self.K)
        self.y_valid_oh = keras.utils.to_categorical(self.y_valid, self.K)
        self.y_test_oh = keras.utils.to_categorical(self.y_test, self.K)

        if self.verbose:
            print('Data specification:')
            print('\tDataset type:          ', self.dataset)
            print('\tNumber of classes:     ', self.K)
            print('\tNumber of channels:    ', self.C)
            print('\tTraining data shape:   ', self.x_train.shape)
            print('\tValidation data shape: ', self.x_valid.shape)
            print('\tTest data shape:       ', self.x_test.shape)

    def split_data(self):
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
        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_valid = self.x_valid.astype("float32") / 255.0
        self.x_test  = self.x_test.astype("float32") / 255.0

    def plot(
        self,
        xx = 12,
        yy = 3,
        save_path=None
    ):
        plt.figure(figsize=(18,yy*2))
        cm = 'gray' if self.C==1 else 'viridis'
        for i in range(xx*yy):
            plt.subplot(yy,xx,i+1)
            plt.imshow(self.x_train[i], cmap=cm)
            plt.title('label=%d'%(self.y_train[i]))
            plt.axis('off')
        plt.show()
       
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()