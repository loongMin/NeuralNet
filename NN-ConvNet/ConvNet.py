f"""
# convolution net model:
## layers i:
layers shape: [width, height, channel_i]


## kernels i:
+ convolution:
    kernels shape: [width, height, channel_(i-1), channel_i]
    forward: Z^[i] = W^[i] * Z^[i-1], 
    backward: dW^[i] += dZ^[i] * Z^[i-1], dZ^[i-1] += W^[i] * dZ^[i]
    
+ pooling:
    max        
    mean
             
+ full connected layers:
    


+ input layer 0: [width, height, 3(rgb)]

+ convolution output layer:

## Programming Parameters
+ kernels: 
    convolution: (layers, kernels, kernel_width, kernel_height, channel_i-1)
    pooling: max, mean
+ padding
+ step
+ layers: (layers, layer_width, layer_height, channel)
+ 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./../data/', train=True, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=128, shuffle=True)


def getXY():
    train_batch = enumerate(train_loader)
    batch_idx, (train_imgs, train_labels) = next(train_batch)
    return np.array(train_imgs), np.array(train_labels)


def get_W_b(n_c, n_c_pre, f_h, f_w):
    W = np.random.rand(n_c, n_c_pre, f_h, f_w)
    W = W / W.sum()
    b = np.random.rand(n_c, 1, 1, 1)
    b = b / b.sum()
    return W, b


def relu(Z):
    A = Z.copy()
    A[A < 0] = 0
    return A


def relu_(Z, dA):
    dZ = dA.copy()
    dZ[Z<0] = 0
    return dZ


softmax_layer_sum = 0
def softmax(Z):
    Z = Z - Z.mean()
    A = np.exp(Z)
    softmax_layer_sum = A.sum()
    return A/softmax_layer_sum


def softmax_(Z):
    Z = Z - Z.mean()
    ex = np.exp(Z)
    return ex*(softmax_layer_sum-ex)/softmax_layer_sum**2


def cross_lose(A, y):
    assert(A.shape == y.shape)
    return -y*np.log(A)


def cross_lose_(A, y):
    return (-1/A)*y


def get_W_b(n_c, n_c_pre, f_h, f_w):
    W = np.random.rand(n_c, n_c_pre, f_h, f_w)
    b = np.random.rand(n_c, 1, 1, 1)
    return W, b


def get_mask(slice, mode):
    if mode=="max":
        mask = slice == slice.max(axis=(1, 2), keepdims=True)
    elif mode=="mean":
        mask = np.ones(slice.shape)
        mask = mask/mask.sum(axis=(1, 2), keepdims=True)
    return mask


def conv_forward(A_prev, W, b, pad, stride, mode="conv"):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (n_C, n_C_prev, f, f)
    b -- Biases, numpy array of shape (n_C, 1, 1, 1)
    stride --
    pad --
    mode --

    Returns:
    Z -- conv output, numpy array of shape (m, n_C, n_H, n_W)
    A --
    """

    # get shape
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    (n_C, n_C_prev, f, f) = W.shape

    # create container Z and padded A_prev
    n_H = (n_H_prev + 2 * pad - f) // stride + 1
    n_W = (n_W_prev + 2 * pad - f) // stride + 1
    Z = np.zeros((m, n_C, n_H, n_W))
    A = np.zeros((m, n_C, n_H, n_W))
    A_prev = np.pad(A_prev, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0))

    # propagation
    for i in range(0, m):
        for j in range(0, n_C):
            for k in range(0, n_H):
                for l in range(0, n_W):
                    assert (k*stride+f-1 < n_H_prev+2*pad and l*stride+f-1 <= n_W_prev+2*pad)
                    A_pre_slice = A_prev[i, :, k*stride:k*stride+f, l*stride:l*stride+f]
                    if mode == "conv":
                        conv = A_pre_slice * W[j] + b[j]
                        Z[i, j, k, l] = conv.sum()
                    elif mode == "max" or mode == "mean":
                        mask = get_mask(A_pre_slice, mode)
                        conv = A_pre_slice * mask
                        A[i, j, k, l] = conv.sum()

    if mode == "conv": A = relu(Z)
    return Z, A


def conv_backward(dA, Z, W, b, A_prev, stride, pad, mode):
    # get shape
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    (n_C, n_C_prev, f, f) = W.shape
    (m, n_C, n_H, n_W) = Z.shape

    # set gradient container
    dZ = relu_(Z, dA)
    dW = np.zeros((n_C, n_C_prev, f, f))
    db = np.zeros((n_C, 1, 1, 1))
    A_prev = np.pad(A_prev, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0))
    dA_prev = np.zeros(A_prev.shape)

    # propagation
    for i in range(0, m):
        for j in range(0, n_C):
            for k in range(0, n_H):
                for l in range(0, n_W):
                    A_pre_slice = A_prev[i, :, k * stride:k * stride + f, l * stride:l * stride + f]
                    if mode == "conv":
                        dW[j] += dZ[i, j, k, l] * A_pre_slice
                        db[j] += dZ[i, j, k, l]
                        dA_prev[i, :, k * stride:k * stride + f, l * stride:l * stride + f] += W[j] * dZ[i, j, k, l]
                    elif mode == "max" or mode == "mean":
                        mask = get_mask(A_pre_slice, mode)
                        dA_prev[i, :, k * stride:k * stride + f, l * stride:l * stride + f] += mask * dZ[i, j, k, l]

    return dW, db, dA_prev[:, :, pad:pad + n_H_prev, pad:pad + n_W_prev]


def fc_forward(A_prev, W, b, g):
    Z = np.dot(W, A_prev) + b
    A = g(Z)
    return Z, A


def fc_backward(A_prev, W, g_, Z, dA):
    n_l, m = Z.shape

    dZ = g_(Z) * dA
    dW = np.dot(dZ, A_prev.T) / m
    db = dA.sum(axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dW, db, dA_prev


def train(batchs, itrs, learning_rate):
    los = []

    # model data
    (A1, Z1, pad1, stride1, mode1, (W1, b1)) = ([], [], 1, 1, "conv", get_W_b(16, 1, 3, 3))
    (A2, Z2, pad2, stride2, mode2, (W2, b2)) = ([], [], 1, 2, "conv", get_W_b(16, 16, 3, 3))
    (A3, Z3, pad3, stride3, mode3, (W3, b3)) = ([], [], 1, 2, "max", get_W_b(16, 16, 3, 3))
    (A4, Z4, pad4, stride4, mode4, (W4, b4)) = ([], [], 0, 2, "conv", get_W_b(16, 16, 3, 3))
    (A5, Z5, pad5, stride5, mode5, (W5, b5)) = ([], [], 0, 1, "conv", get_W_b(32, 16, 3, 3))
    W6 = np.random.rand(16, 32)
    b6 = np.random.rand(16, 1)
    W7 = np.random.rand(10, 16)
    b7 = np.random.rand(10, 1)

    for i in range(0, batchs):
        A0, labels = getXY()
        y = np.array([np.array(labels).tolist()])
        softmax_y = np.zeros((10, y.shape[1]))
        for i in range(0, y.shape[1]):
            softmax_y[y[0, i], i] = 1

        for itr in range(0, itrs):
            print(itr, "================================================================================")
        # forward propagation   conv_forward(A_prev, W, b, pad, stride, mode="conv")
            # conv
            Z1, A1 = conv_forward(A0, W1, b1, pad1, stride1, "conv")
            Z2, A2 = conv_forward(A1, W2, b2, pad2, stride2, "conv")
            Z3, A3 = conv_forward(A2, W3, b3, pad3, stride3, "max")
            Z4, A4 = conv_forward(A3, W4, b4, pad4, stride4, "conv")
            Z5, A5 = conv_forward(A4, W5, b5, pad5, stride5, "conv")

            # conv to fc
            m, A5_c, A5_h, A5_w = A5.shape
            A5_ = A5.flatten('C').reshape((m, -1), order='C').T

            # fc
            Z6, A6 = fc_forward(A5_, W6, b6, relu)
            Z7, A7 = fc_forward(A6, W7, b7, softmax)

        # lose
            l = cross_lose(A7, softmax_y)
            los.append(l)
            print(l)

        # backward propagation
            # fc fc_backward(A_prev, W, g_, Z, dA):
            dA7 = l * cross_lose_(A7, softmax_y)
            dW7, db7, dA6 = fc_backward(A6, W7, softmax, Z7, dA7)
            dW6, db6, dA5_ = fc_backward(A5_, W6, relu, Z6, dA6)

            # fc to conv
            dA5 = dA5_.reshape(A5.shape, order='C')

            # conv conv_backward(dA, Z, W, b, A_prev, stride, pad, mode)
            dW5, db5, dA4 = conv_backward(dA5, Z5, W5, b5, A4, stride5, pad5, mode="conv")
            dW4, db4, dA3 = conv_backward(dA4, Z4, W4, b4, A3, stride4, pad4, mode="conv")
            dW3, db3, dA2 = conv_backward(dA3, Z3, W3, b3, A2, stride3, pad3, mode="max")
            dW2, db2, dA1 = conv_backward(dA2, Z2, W2, b2, A1, stride2, pad2, mode="conv")
            dW1, db1, dA0 = conv_backward(dA1, Z1, W1, b1, A0, stride1, pad1, mode="conv")

        # refresh the weights
            W7 = W7 - learning_rate * dW7
            b7 = b7 - learning_rate * b7
            W6 = W6 - learning_rate * dW6
            b6 = b6 - learning_rate * b6

            W5 = W5 - learning_rate * dW5
            b5 = b5 - learning_rate * b5
            W4 = W4 - learning_rate * dW4
            b4 = b4 - learning_rate * b4
            W3 = W3 - learning_rate * dW3
            b3 = b3 - learning_rate * b3
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * b2
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * b1

    plt.plot(los)
    plt.show()

if __name__ == '__main__':
    train(1, 100, 0.01)


def test():
    A_prev = np.array([[
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ],
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ],
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]
    ]])
    print(A_prev.shape)

    W = np.array([
        [
            [
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [1, 0, -1],
                [1, 0, -1]
            ]
        ],
        [
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ]
        ]
    ])
    print(W.shape)
    b = np.array([
        [
            [
                [1]
            ]
        ],
        [
            [
                [2]
            ]
        ]
    ])
    print(b.shape)

    Z, A = conv_forward(A_prev, W, b, 1, 1)
    print(Z.shape)
    print(Z)
    print("-----------------")
    print(A)

