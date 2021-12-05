import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
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
    
    
        
#lose function 
    binary-classification:
        L(a, y) = - (y·ln(a) + (1-y)ln·(1-a))
        L'(a) = - (y/a + (1-y)/(1-a))

    regression:
        L(a, y) = EMS(a, y) = (a-y)^2
        L'(a) = 2|a-y|
'''


'''
# batch normalization 
    forward:
    
    backward:

    
    hyper parameters:
        #learning rate: a
        #activacation: g
        #number of examples: m
        #units: n, nx = n_0
        #number of layer: l
        #lose function: L
'''


class NeuralNet:
    # parameters
    G_list = []
    A_list = []
    Z_list = []
    W_list = []
    b_list = []

    loseFunction = ""  # lose function type
    y = []  # target y

    # hyperparameters
    l = 0  # number of layers
    n_list = []  # number of units in every layer
    m = 16  # m exmples in training
    a = 0.03  # learning rate
    itr = 128  # times of iteration

    # train state record
    lose_list = []  # lose of mean in every iteration

    # register the activation function and its derivative to dictionary actGdic
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_(z):
        a = 1 / (1 + np.exp(-z))
        return a * (1 - a)

    def relu(z):
        return max(0, z)

    def relu_(z):
        return int(z >= 0)

    def leakyRelu(z):
        return max(0.01 * z, z)

    def leakyRelu_(z):
        return max(0.01, int(z >= 0))

    actiGdic = {"sigmoid": sigmoid,
                "sigmoid_": sigmoid_,
                "relu": relu,
                "relu_": relu_,
                "leakyRelu": leakyRelu,
                "leakyRelu_": leakyRelu_}

    # register lose function and its derivative to dictionary loseLdic
    def L_binaryClassfication(a, y):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def L_binaryClassfication_(a, y):
        return -(y / a + (1 - y) / (1 - a))

    def L_regression(a, y):
        return (a - y) ** 2

    def L_regression_(a, y):
        return 2 * (a - y)

    loseLdic = {"L_b": L_binaryClassfication,
                "L_b_": L_binaryClassfication_,
                "L_r": L_regression,
                "L_r_": L_regression_}

    #
    def __init__(self):
        self.l = 0

    def load_data_piece(self, A_0, y):
        self.n_list.append(A_0.shape[0])
        self.m = A_0.shape[1]
        self.y = y.tolist()

    def creat_neural_network(self):
        self.G_list.append("")
        self.Z_list.append(np.array([]))
        self.A_list.append(self.a_random_array(self.n_list[0], self.m))
        self.W_list.append(np.array([]))
        self.b_list.append(np.array([]))

        #
        pattern_layer = re.compile(r"(relu|leakyrelu|sigmoid)\s*(\d+)")
        while True:
            str = input("activation(relu, leakyrelu, sigmoid) with a number of units:")
            if (str == "end"): break
            match = pattern_layer.match(str)
            if match:
                n_l = int(match.group(2))
                self.n_list.append(n_l)
                self.l += 1

                self.G_list.append(match.group(1))
                self.Z_list.append(self.a_random_array(n_l, self.m))
                self.A_list.append(self.a_random_array(n_l, self.m))
                self.W_list.append(self.a_random_array(n_l, self.n_list[self.l - 1], 0.001))
                self.b_list.append(self.a_random_array(n_l, 1, 0.001))
            else:
                print("pattern inputted is wrong!, re-input for this network layer.")

        #
        pattern_lose = re.compile(r"(L_b|L_r)")
        while True:
            str = input("chose a lose function(L_r, L_b):")
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
            Z = np.array(self.Z_list[i])
            A = np.array(self.A_list[i])
            W = np.array(self.W_list[i])
            b = np.array(self.b_list[i])
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

    # training part
    def piece_train_network(self, itr, a):
        for i in range(0, itr):
            self.forward()
            lose_itr = self.lose()
            self.lose_list.append(lose_itr.sum() / self.m)

            self.backward(a, lose_itr)
            print("The ", i, "th propagation!")

    def forward(self):
        A_front = np.array(self.A_list[0])
        for i in range(1, self.l + 1):
            W = np.array(self.W_list[i])
            b = np.array(self.b_list[i])

            # forward propagation
            Z = W.dot(A_front) + b
            A = np.array(list(map(self.actiGdic[self.G_list[i]], Z.flatten('C')))
                         ).reshape(self.n_list[i], -1)

            # refressh the A and Z
            self.Z_list[i] = Z.tolist()
            self.A_list[i] = A.tolist()
            A_front = A

    def lose(self):
        y_hat = np.array(self.A_list[self.l])
        y = np.array(self.y)
        if y_hat.shape == y.shape:
            return np.array(list(map(self.loseLdic[self.loseFunction],
                                     y_hat.flatten('C'),
                                     y.flatten('C')))
                            ).reshape(self.n_list[self.l], -1)
        else:
            print("the last layer did not pattern the format of y")
            exit(-1)

    def backward(self, a, lose):
        dA = lose * np.array(list(map(self.loseLdic[self.loseFunction + "_"],
                                      np.array(self.A_list[self.l]).flatten('C'),
                                      np.array(self.y).flatten('C')))
                             ).reshape(self.n_list[self.l], -1)
        for i in range(self.l, 0, -1):
            Z = np.array(self.Z_list[i])
            W = np.array(self.W_list[i])
            b = np.array(self.b_list[i])
            A_1 = np.array(self.A_list[i - 1])

            # backward propagation
            dZ = dA * np.array(list(map(self.actiGdic[self.G_list[i] + "_"], Z.flatten('C')))
                               ).reshape(self.n_list[i], -1)
            dW = np.dot(dZ, A_1.T) / self.m
            db = np.sum(dZ, axis=1, keepdims=True) / self.m
            dA = np.dot(W.T, dZ)

            if i == 1:
                print("---------------------------------------------------------------")
                print(dW)
            # refresh the W and b
            W = W - a * dW
            b = b - a * db
            self.W_list[i] = W.tolist()
            self.b_list[i] = b.tolist()

    def show_lose(self):
        plt.plot(self.lose_list)
        plt.show()


def house_price():
    neuralNet = NeuralNet()
    df_x = pd.read_csv("./data/house_price/test.csv")
    df_y = pd.read_csv("./data/house_price/price.csv")
    features = ['MSSubClass', 'LotFrontage', 'LotArea']
    x = np.array(df_x[features].T)
    y = np.array([np.array(df_y['SalePrice'].T).tolist()])

    neuralNet.load_data_piece(x, y)
    neuralNet.creat_neural_network()
    neuralNet.piece_train_network(10, 0.3)
    neuralNet.show_lose()


def minist_hand_writing(batch_size_train, batch_size_test):
    neuralNet = NeuralNet()
    import torch.utils.data as Data
    import torchvision
    train_loader = Data.DataLoader(
        torchvision.datasets.MNIST('./../data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = Data.DataLoader(
        torchvision.datasets.MNIST('./../data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)


if __name__ == '__main__':
    house_price()



