import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from PIL import Image
import json
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
''' FUNCTIONS
#Activation function:
    ReLu
        forward: g(z) = max(0, z)
        backward: g'(z) = int(z>=0)
        
    leaky Relu
        forward: g(z) = max(0.01z, z)
        backward: g'(z) = max(int(z>=0), 0.01)
        
    sigmoid
        forward: a = g(z) = 1/(1+e^(-z));
        backward: g'(z) = a(1-a)
    
    softmax
        define sumA as sum of softmax layer in output A， maxZ as max of softmax layer in input Z
        forward: a = g(z) = (1/e^(z-maxZ))/sum
        backward: g'(z) = -(sum-e^t)*e^t / sum^2
        
#lose function 
    binary-classification
        L(a, y) = - (y·ln(a) + (1-y)ln·(1-a))
        L'(a) = - (y/a + (1-y)/(1-a))
    
    multiple-classification: softmax
        L(a, y) = Epsilon{-y*log(a)}
        L'(a, y) = {0, y=0; 1/a, y=1}
        
    regression
        L(a, y) = EMS(a, y) = (a-y)^2
        L'(a) = 2|a-y|
'''


''' PARAMETERS DEFINATION
#layer model:
    forward:
        Z = W · A_1 + b
        A = g(Z)
    backward:
        dZ = dA * f'(Z)
        dW = dZ · A_1.T
        db = np.sum(dZ, axis=1, keepdim =true) 
        dA_1 = W.T · dZ
    
    
    parameters:
        W, b Z A
    
    hyper parameters:
        #learning rate: a
        #activacation: g
        #number of examples: m
        #units: n, nx = n_0
        #number of layer: l
        #lose function: L
'''


class NeuralNet:
    # compiling parameters
    np_dtype = 'float64'
    debug = False

    # parameters
    G_list = []
    A_list = []
    Z_list = []
    W_list = []
    b_list = []

    loseFunction = ""           # lose function type
    y = []                      # target y

    # hyperparameters
    l = 0                       # number of layers
    n_list = []                 # number of units in every layer

    # train state record

    m = 16                      # m examples for each batch
    a = 0.03                    # learning rate
    itr = 128                   # times of iteration
    batch_lose_list = []        # lose of mean in every iteration
    train_acc = []
    dev_acc = []

    #
    # register the activation function and its derivative to dictionary actGdic
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_(z):
        a = 1 / (1 + np.exp(-z))
        return a*(1-a)

    def relu(z):
        return max(0, z)

    def relu_(z):
        return int(z >= 0)

    def leakyRelu(z):
        return max(0.01 * z, z)

    def leakyRelu_(z):
        return max(0.01, int(z >= 0))

    def softmax(z, maxZ):
        return np.exp(z-maxZ)

    def softmax_(z, sumA):
        ex = np.exp(z)
        return ex * (sumA - ex) / (sumA ** 2)

    actiGdic = {"sigmoid": sigmoid,
                "sigmoid_": sigmoid_,
                "relu": relu,
                "relu_": relu_,
                "leakyRelu": leakyRelu,
                "leakyRelu_": leakyRelu_,
                "softmax": softmax,
                "softmax_": softmax_}

    # register lose function and its derivative to dictionary loseLdic
    def L_binaryClassfication(a, y):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def L_binaryClassfication_(a, y):
        return -(y / (a + 0.1**8) + (1 - y) / (1 - a + 0.1**8))

    def L_regression(a, y):
        return (a - y)**2

    def L_regression_(a, y):
        return 2 * (a - y)

    def L_cross_entropy(a, y):
        return -y*np.log(a)

    def L_cross_entropy_(a, y):
        return -y/(a + 0.1**8)

    loseLdic = {"L_b": L_binaryClassfication,
                "L_b_": L_binaryClassfication_,
                "L_r": L_regression,
                "L_r_": L_regression_,
                "L_c": L_cross_entropy,
                "L_c_": L_cross_entropy_}

    #
    def __init__(self):
        self.l = 0

    def load_data_piece(self, A_0, y=np.array([])):
        if len(self.A_list) == 0:
            self.A_list.append(A_0)
            self.n_list.append(A_0.shape[0])
        else:
            self.A_list[0] = A_0.tolist()
            self.n_list[0] = A_0.shape[0]
        self.y = y.tolist()
        self.m = A_0.shape[1]

    def creat_neural_network(self, n_x=0, m=0):
        """
        if you give the m_x and m of input layer, the net will begin with it, so the X's shape must satisfy your givens
        :param n_x:
        :param m:
        :return:
        """
        self.G_list.append("")
        if len(self.n_list) == 0:
            self.A_list.append(np.zeros((n_x, m), dtype=self.np_dtype).tolist())
            self.n_list.append(n_x)
            self.m = m
        self.Z_list.append([])
        self.W_list.append([])
        self.b_list.append([])
        pattern_layer = re.compile(r"(relu|leakyRelu|sigmoid|softmax)\s*(\d+)")
        while True:
            str = input("activation(relu, leakyrelu, sigmoid, softmax) with a number of units:")
            if str=="end": break
            match = pattern_layer.match(str)
            if match:
                n_l = int(match.group(2))
                self.n_list.append(n_l)
                self.l += 1
                self.G_list.append(match.group(1))
                self.W_list.append(self.a_random_array(n_l, self.n_list[self.l - 1], 0.001))
                self.b_list.append(self.a_random_array(n_l, 1, 0.001))
            else:
                print("pattern inputted is wrong!, re-input for this network layer.")

        #
        pattern_lose = re.compile(r"(L_b|L_r|L_c)")
        while True:
            str = input("chose a lose function(L_r, L_b, L_c):")
            match = pattern_lose.match(str)
            if match:
                self.loseFunction = match.group(1)
                break
            else:
                print("pattern inputted is wrong!")

    def a_random_array(self, row, col=1, mul=1):
        return np.random.rand(row, col) * mul

    def show_all_parameters_pattern(self):
        for i in range(0, self.l + 1):
            print("======================================================================")
            print("The ", i, "th", "layer with ", self.n_list[i], "units as below:")
            Z = np.array(self.Z_list[i], dtype=self.np_dtype)
            A = np.array(self.A_list[i], dtype=self.np_dtype)
            W = np.array(self.W_list[i], dtype=self.np_dtype)
            b = np.array(self.b_list[i], dtype=self.np_dtype)
            print("G:", self.G_list[i])
            print("Z", Z.shape)
            # print(Z)
            print("A:", A.shape)
            # print(A)
            print("W", W.shape)
            # print(W)
            print("b", b.shape)
            # print(b)
        print("======================================================================")
        print("y:", np.array(self.y).shape)

    # batch training
    def piece_train_network(self, itr, a):
        for i in range(0, itr):
            self.forward()
            lose_itr = self.lose()
            self.batch_lose_list.append(lose_itr.sum() / self.m)
            self.backward(a, lose_itr)

    def forward(self):
        del self.A_list[1:]
        del self.Z_list[1:]
        A_front = np.array(self.A_list[0], dtype=self.np_dtype)
        for i in range(1, self.l + 1):
            W = np.array(self.W_list[i], dtype=self.np_dtype)
            b = np.array(self.b_list[i], dtype=self.np_dtype)

            # A_1 forward propagate to Z
            Z = W.dot(A_front) + b

            # Z forward propagate to A
            A = []
            if self.G_list[i] == "softmax":                                 # softmax layer
                maxZ =  Z.max(axis=0)
                A = np.array(list(map(self.actiGdic[self.G_list[i]],
                                      Z.flatten('C'),
                                      maxZ.repeat(self.n_list[i]))),
                             dtype=self.np_dtype
                             ).reshape(self.n_list[i], -1)
            else:                                                           # relu, leakyRelu, sigmoid, tanh
                A = np.array(list(map(self.actiGdic[self.G_list[i]], Z.flatten('C'))),
                             dtype=self.np_dtype
                             ).reshape(self.n_list[i], -1)

            # refressh the A and Z with results
            self.Z_list.append(Z.tolist())
            self.A_list.append(A.tolist())
            A_front = A

            # debug
            if i == self.l and self.debug:
                print("=====================================================================================================================================")
                print("W:", W.shape, W.min(), W.mean(), W.max(), W.sum())
                print("Z:", Z.shape, Z.min(), Z.mean(), Z.max(), Z.sum())
                print("A:", A.shape, A.min(), A.mean(), A.max(), A.sum())

    def lose(self):
        y_hat = np.array(self.A_list[self.l], dtype=self.np_dtype)
        y = np.array(self.y, dtype=self.np_dtype)
        if y_hat.shape == y.shape:
            return np.array(list(map(self.loseLdic[self.loseFunction],
                                 y_hat.flatten('C'),
                                 y.flatten('C'))),
                            dtype=self.np_dtype
                            ).reshape(self.n_list[self.l], -1)
        else:
            print("the last layer did not pattern the format of y")
            exit(-1)

    def backward(self, a, lose):
        dA = lose * np.array(list(map(self.loseLdic[self.loseFunction+"_"],
                                      np.array(self.A_list[self.l]).flatten('C'),
                                      np.array(self.y).flatten('C'))),
                             dtype=self.np_dtype
                             ).reshape(self.n_list[self.l], -1)
        for i in range(self.l, 0, -1):
            Z = np.array(self.Z_list[i], dtype=self.np_dtype)
            W = np.array(self.W_list[i], dtype=self.np_dtype)
            b = np.array(self.b_list[i], dtype=self.np_dtype)
            A_1 = np.array(self.A_list[i-1], dtype=self.np_dtype)

            # dA backward propagate to Z
            dZ = []
            if self.G_list[i] == "softmax":
                sumA = np.array(self.A_list[i]).sum(axis=0, dtype=self.np_dtype)
                dZ = dA * np.array(list(map(self.actiGdic[self.G_list[i] + "_"],
                                            Z.flatten('C'),
                                            sumA.repeat(self.n_list[i]))),
                                   dtype=self.np_dtype
                                   ).reshape(self.n_list[i], -1)
            else:
                dZ = dA * np.array(list(map(self.actiGdic[self.G_list[i]+"_"], Z.flatten('C'))),
                                   dtype=self.np_dtype
                                   ).reshape(self.n_list[i], -1)

            # dZ backward propagate to W and b
            dW = np.dot(dZ, A_1.T) / self.m
            db = np.sum(dZ, axis=1, keepdims=True) / self.m
            dA = np.dot(W.T, dZ)

            # refresh the W and b with results of dW and db
            W = W - a * dW
            b = b - a * db
            self.W_list[i] = W.tolist()
            self.b_list[i] = b.tolist()

            if self.debug:
                print("--", i, "th dW -----------------------------------------------------------------------")
                print("dW:", dW.shape, dW.min(), dW.mean(), dW.max(), dW.sum())





    #



    def record_train_acc_softmax(self):
        y_hat = np.array(self.A_list[self.l])
        y = np.array(self.y)
        y_predict = y_hat.argmax(axis=0)
        self.train_acc.append((y == y_predict).sum()/y.shape[1])
        return

    def show_train_dev_loss(self):
        plt.plot(range(1, len(self.batch_lose_list)+1), self.batch_lose_list, )
        plt.show()

    def show_acc(self):
        plt.plot(range(1, len(self.train_acc)+1), self.train_acc, '-')
        # plt.plot(range(1, len(self.dev_acc)), self.dev_acc, '-')
        plt.show()








    # dev Testing
    def detect(self, x):
        A_0 = np.zeros((self.n_list[0], self.m), dtype=self.np_dtype)
        A_0[:, 0] = x[:, 0]
        self.A_list[0] = A_0
        self.forward()
        return self.A_list[self.l]

    def save_weights(self):
        date = time.strftime("%Y-%m-%d %H:%M:%S" + time.localtime())
        with open('./weights/'+date+'.json', 'w') as weights_json_F:
            weights = {
                'G_list': self.G_list,
                'W_list': self.W_list,
                'b_list': self.b_list,
                'loseFunction': self.loseFunction,
                'l': self.l,
                'n_list': self.n_list,
            }
            weights_json = json.dumps(weights)
            weights_json_F.write(weights_json)
            weights_json_F.close()

    def load_weights(self, date):
        with open('./weights/' + date + '.json', 'w') as weights_json_F:
            weights_json = weights_json_F.read()
            weights = json.loads(weights_json)
            self.G_list = weights['G_list']
            self.W_list = weights['W_list']
            self.b_list = weights['b_list']
            self.loseFunction = weights['loseFunction']
            self.l = weights['l']
            self.n_list = weights['n_list']
            weights_json_F.close()


def house_price():
    neuralNet = NeuralNet()
    df_x = pd.read_csv("./data/house_price/test.csv")
    df_y = pd.read_csv("./data/house_price/price.csv")
    features = ['MSSubClass', 'LotFrontage', 'LotArea']
    x = np.array(df_x[features].T,)
    y = np.array([np.array(df_y['SalePrice'].T).tolist()])

    neuralNet.load_data_piece(x, y)
    neuralNet.creat_neural_network()
    neuralNet.piece_train_network(10, 0.3)
    neuralNet.show_train_dev_loss()


def minist_hand_writing():
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
        batch_size=1, shuffle=True)

    train_batch = enumerate(train_loader)
    neuralNet = NeuralNet()
    neuralNet.creat_neural_network(28*28, 128)
    neuralNet.debug = True

    for i in range(0, 100):
        batch_idx, (train_imgs, train_labels) = next(train_batch)
        img_shape = np.array(train_imgs).shape

        x = np.array(train_imgs).flatten('C').reshape(img_shape[2]*img_shape[3], -1, order='C')
        y = np.array([np.array(train_labels).tolist()])
        softmax_y = np.zeros((10, y.shape[1]))
        for i in range(0, y.shape[1]):
            softmax_y[y[0, i], i] = 1
        neuralNet.load_data_piece(x, softmax_y)

        with open("train_controlling.json", 'r') as conF:
            conStr = conF.read()
            conJson = json.loads(conStr)

            if conJson["save_weights_itr"] == i:
                neuralNet.save_weights()
            if conJson["continue_learning"] == 1:
                neuralNet.piece_train_network(conJson["batch_itr"], conJson["learning_rate"])
                neuralNet.record_train_acc_softmax()

            else:
                break
            conF.close()

    neuralNet.show_train_dev_loss()
    neuralNet.show_acc()


    for i in range(0, 64):
        cmd = input("========================================================================================")
        if cmd == "save_weigths":
            neuralNet.save_weights()
        elif cmd == "load_weigths":
            date = input("date")
            neuralNet.load_weights(date)
        else:
            y = np.array(neuralNet.y)
            y_hat = np.array(neuralNet.A_list[neuralNet.l][i])
            print(y.argmax(axis=0))
            print(y_hat.argmax(axis=0))


if __name__ == '__main__':
    test()