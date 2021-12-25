import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def sigmoid(Z):
    return 1. / (1. + np.exp(-Z))

def sigmoid_(Z):
    A = 1. / (1. + np.exp(-Z))
    return A*(1.-A)

def relu(Z):
    Z[Z<0] = 0.
    return Z

def relu_(Z):
    Z[Z>=0] = 1.
    Z[Z<0] = 0.
    return Z

def leakyRelu(Z):
    Z[Z<0] = 0.01 * Z[Z<0]
    return Z

def leakyRelu_(Z):
    Z[Z>=0] = 1.
    Z[Z<0] = 0.01
    return Z

def softmax(Z):
    A = np.exp(Z-Z.mean(axis=0, keepdims=True))
    softmax_sum = A.sum(axis=0, keepdims=True)
    return A / softmax_sum

def softmax_(Z):
    A = np.exp(Z-Z.mean())
    softmax_sum = A.sum(axis=0, keepdims=True)
    return A * (softmax_sum - A) / (softmax_sum ** 2)

def L_binaryClassfication(A, Y):
    return -(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def L_binaryClassfication_(A, Y):
    return -(Y / (A + 0.1**8) + (1 - Y) / (1 - A + 0.1**8))

def L_cross_entropy(A, Y):
    return -Y*np.log(A + 0.1**8)

def L_cross_entropy_(A, Y):
    return -Y/(A + 0.1**8)


def forward(A_prev, W, b, g):
    Z = W.dot(A_prev) + b
    A= g(Z)
    return Z, A


def backward(A_prev, W, Z, g_, dA):
    dZ = dA * g_(Z)
    dW = dA.dot(A_prev.T) / Z.shape[1]
    db = dA.sum(axis=1, keepdims=True) / Z.shape[1]
    dA_prev = W.T.dot(dA)
    return dA_prev, dW, db, dZ


def get_pars(n_pre, n):
    W = np.random.rand(n, n_pre)
    W = W / W.sum()
    b = np.random.rand(n, 1)
    b = b / b.sum()
    Z = []
    A = []
    dW = []
    db = []
    dZ = []
    dA = []
    return W, b, Z, A, dW, db, dZ, dA


# batch training
def train(getXY, itr, a):
    # define net structure and initialize parameters
    m = 128
    W1, b1, Z1, A1, dW1, db1, dZ1, dA1 = get_pars(28 * 28, 28)
    W2, b2, Z2, A2, dW2, db2, dZ2, dA2 = get_pars(28, 16)
    W3, b3, Z3, A3, dW3, db3, dZ3, dA3 = get_pars(16, 10)

    train_loss = []
    dev_loss = []

    for i in range(0, itr):
        trainA0, trainY, devA0, devY = getXY()

        # train
        # forward
        A0 = trainA0
        Z1, A1 = forward(A0, W1, b1, relu)
        Z2, A2 = forward(A1, W2, b2, relu)
        Z3, A3 = forward(A2, W3, b3, softmax)

        # get loss
        los = L_cross_entropy(A3, trainY)
        train_loss.append(los.sum() / m)

        # backward propagation
        dA3 = los * L_cross_entropy_(A3, trainY)
        dA2, dW3, db3, dZ3 = backward(A2, W3, Z3, softmax_, dA3)
        dA1, dW2, db2, dZ2 = backward(A1, W2, Z2, relu_, dA2)
        dA0, dW1, db1, dZ1 = backward(A0, W1, Z1, relu_, dA1)

        print(i, "=================================================================")
        print(b3.T)

        # refresh the data
        W1 = W1 - a * dW1
        b1 = b1 - a * db1

        W2 = W2 - a * dW2
        b2 = b2 - a * db2

        W3 = W3 - a * dW3
        b3 = b3 - a * db3

    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.show()


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./../data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=128, shuffle=True)


test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./../data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=128, shuffle=True)


def reshape(imgs, labels):
    img_shape = np.array(imgs).shape
    x = np.array(imgs).flatten('C').reshape(-1, img_shape[2] * img_shape[3], order='C').T
    x = x / x.sum()
    y = np.array([np.array(labels).tolist()])
    softmax_y = np.zeros((10, y.shape[1]))
    for i in range(0, y.shape[1]):
        softmax_y[y[0, i], i] = 1
    return x, softmax_y


def getXY():
    train_batch = enumerate(train_loader)
    batch_idx, (train_imgs, train_labels) = next(train_batch)
    train_x, train_y = reshape(train_imgs, train_labels)

    test_batch = enumerate(test_loader)
    batch_idx, (test_imgs, test_labels) = next(test_batch)
    test_x, test_y = reshape(test_imgs, test_labels)

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train(getXY, 10000, 0.003)














def house_price():
    neuralNet = NeuralNet()
    df_x = pd.read_csv("./data/house_price/test.csv")
    df_y = pd.read_csv("./data/house_price/price.csv")
    features = ['MSSubClass', 'LotFrontage', 'LotArea']
    x = np.array(df_x[features].T,)
    y = np.array([np.array(df_y['SalePrice'].T).tolist()])








