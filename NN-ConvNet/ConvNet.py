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


def relu(Z):
    A = Z.copy()
    A[A < 0] = 0
    return A

def conv_forward(A_prev, W, b, stride, pad):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_C_prev, n_H_prev, n_W_prev)
    W -- Weights, numpy array of shape (n_C, n_C_prev, f, f)
    b -- Biases, numpy array of shape (n_C, 1, 1, 1)
    stride --
    pad --

    Returns:
    Z -- conv output, numpy array of shape (m, n_C, n_H, n_W)
    cache -- cache of values needed for the conv_backward() function
    """

    # get shape
    (m, n_C_prev, n_H_prev, n_W_prev) = A_prev.shape
    (n_C, n_C_prev, f, f) = W.shape

    # create container Z and padded A_prev
    n_H = int(np.floor((n_H_prev + 2 * pad - f) / stride + 1))
    n_W = int(np.floor((n_W_prev + 2 * pad - f) / stride + 1))
    Z = np.zeros((m, n_C, n_H, n_W))
    A = np.zeros((m, n_C, n_H, n_W))
    A_prev = np.pad(A_prev, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0))

    for i in range(0, m):
        for j in range(0, n_C):
            for k in range(0, n_H):
                for l in range(0, n_W):
                    assert (k*stride+f-1 < n_H_prev+2*pad and l*stride+f-1 < n_W_prev+2*pad)
                    A_pre_conv_win = A_prev[i, :, k*stride:k*stride+f, l*stride:l*stride+f]
                    conv = A_pre_conv_win * W[j] + b[j]
                    Z[i, j, k, l] += conv.sum()
    assert (Z.shape == (m, n_C, n_H, n_W))
    A = relu(Z)
    return Z, A



























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


def get_W_b(n_c, n_c_pre, f_h, f_w):
    W = np.random.rand(n_c, n_c_pre, f_h, f_w)
    b = np.random.rand(n_c, 1, 1, 1)
    return (W, b)


if __name__ == '__main__':
    (W, b) = get_W_b(16, 1, 3, 3)
    print(W)

