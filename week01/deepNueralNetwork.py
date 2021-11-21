import numpy as np;
import matplotlib.pyplot as plt
import re
import pandas as pd

'''
Activation function:
    ReLu
        forward: g(z) = max(0, z)
        backward: g'(z) = int(z>=0)
        
    leaky Relu
        forward: g(z) = max(0.01z, z)
        backward: g'(z) = max(int(z>=0), 0.01)
        
    sigmoid
        forward: a = g(z) = 1/(1+e^(-z));
        backward: g'(z) = a(1-a)
        
        
lose function for binary-classification
    L(a, y) = - (y·ln(a) + (1-y)ln·(1-a))
    L'(a) = - (y/a + (1-y)/(1-a))

lose function for regression
    L(a, y) = EMS(a, y) = (a-y)^2
    L'(a) = 2|a-y|

'''

'''
layer model:
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
        #units: n
        #number of layer: x
        #lose function: L
'''

'''


'''


class NeuralNet:
    # parameters
    G_list = []
    A_list = []
    Z_list = []
    W_list = []
    b_list = []

    loseFunction = ""

    # hyperparameters
    x = 0
    n_list = []
    m = 16
    a = 0.03
    itr = 128

    # train state record
    lose_list = []
    y = []


    # activation function and its derivative
    def sigmoid(z):
        return 1 / (1 + np.e(-z))

    def sigmoid_(self, z):
        a = self.sigmoid()
        return a*(1-a)

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

    # lose function
    def L_binaryClassfication(a, y):
        return -(y * np.log(a) + (1 - y) * np.log(1 - a))

    def L_binaryClassfication_(a, y):
        return -(y / a + (1 - y) / (1 - a))

    def L_regression(a, y):
        return np.sqrt(a - y)

    def L_regression_(a, y):
        return 2 * np.abs(a - y)

    loseLdic = {"L_b": L_binaryClassfication,
                "L_b_": L_binaryClassfication_,
                "L_r": L_regression,
                "L_r_": L_regression_}

    def __init__(self,  m=16):
        self.m = m
        self.x = -1

    def creat_neural_network(self, n_0):
        self.n_list.append(int(n_0))
        self.x += 1

        self.G_list.append("")
        self.Z_list.append(np.array([]))
        self.A_list.append(self.a_random_array(n_0, self.m))
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
                self.x += 1

                self.G_list.append(match.group(1))
                self.Z_list.append(self.a_random_array(n_l, self.m))
                self.A_list.append(self.a_random_array(n_l, self.m))
                self.W_list.append(self.a_random_array(n_l, self.n_list[self.x - 1], 0.001))
                self.b_list.append(self.a_random_array(n_l, 1, 0.001))
            else:
                print("pattern inputted is wrong!, re-input for this network layer.")
                continue

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
        for i in range(0, self.x+1):
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

    #
    def piece_train_network(self, itr, a):
        for i in range(0, itr):
            self.forward()
            lose_itr  = self.lose()
            self.lose_list.append(lose_itr)
            self.backward(a, lose_itr)

    def load_data_piece(self):
        test_A = [
            [2, 4, 2, 3, 4, 6, 3, 6, 7, 10],
            [2, 4, 2, 3, 4, 6, 3, 6, 7, 10],
            [2, 4, 2, 3, 4, 6, 3, 6, 7, 10],
            [2, 4, 2, 3, 4, 6, 3, 6, 7, 10],
            [2, 4, 2, 3, 4, 6, 3, 6, 7, 10]
        ]
        test_y = [13, 21, 4, 6, 2, 4, 7, 2, 6, 6]

        self.A_list[0] = test_A
        self.y = test_y
        return

    def forward(self):
        A_front = np.array(self.A_list[0])
        for i in range(1, self.x+1):
            self.propagaton_position = i;
            W = np.array(self.W_list[i])
            b = np.array(self.b_list[i])

            Z = W.dot(A_front) + b
            A = np.array(list(map(self.actiGdic[self.G_list[i]], Z.flatten('C')))
                         ).reshape(self.n_list[i], -1)
            self.Z_list[i] = Z.tolist()
            self.A_list[i] = A.tolist()
            A_front = A

    def lose(self):
        return np.array(list(map(self.loseLdic[self.loseFunction],
                                 np.array(self.A_list[self.x]).flatten('C'),
                                 np.array(self.y).flatten('C')))
                        ).reshape(self.n_list[self.x], -1)

    def backward(self, a, lose):
        dA = lose * np.array(list(map(self.loseLdic[self.loseFunction+"_"],
                                      np.array(self.A_list[self.x]).flatten('C')
                                      ))
                             ).reshape(self.n_list[self.x], -1)
        for i in range(self.x, 0, -1):
            self.propagaton_position = i
            Z = np.array(self.Z_list[i])
            A = np.array(self.A_list[i])
            W = np.array(self.W_list[i])
            b = np.array(self.b_list[i])

            dZ = dA * np.array(list(map(self.actiGdic[self.G_list[i]+"_"], Z.flatten('C')))
                               ).reshape(self.n_list[i], -1)
            dW = dZ * A.T
            db = np.sum(dZ, axis=1, keepdims=True)

            dA = W.T * dZ

            # refresh the W and b
            W = W - a * dW
            b = b - a * db
            self.W = W.tolist()
            self.b = b.tolist()

    def show_lose(self):
        plt.plot(self.lose_list)









if __name__ == '__main__':
    neuralNet = NeuralNet(10)
    neuralNet.creat_neural_network(5)
    neuralNet.show_all_parameters_pattern()

    neuralNet.load_data_piece()
    neuralNet.piece_train_network(10, 0.03)
    neuralNet.show_lose()









