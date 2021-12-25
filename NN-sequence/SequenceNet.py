''''
one hot encoding: https://blog.csdn.net/dulingtingzi/article/details/51374487
负采样的核心思想是，如果将自然语言看作是一串单词的随机组合，那么它的出现概率是很小的。于是，如果我们将拼凑的单词随机组合（负采样）起来将会以很大的概率不会出现在当前文章中。于是，我们很显然应该至少让我们的模型在这些负采样出来的单词组合上面出现概率应该尽可能地小，同时要让真正出现在文中的单词组合出现概率大
<<<<<<< HEAD
'''

import numpy as np
import pandas as pd

def relu(Z):
    A = Z.copy()
    A[A < 0] = 0
    return A

def relu_(Z, dA):
    dZ = dA.copy()
    dZ[Z<0] = 0
    return dZ

def tanh(Z):
    return Z


def tanh_(Z):
    return Z

def softmax(Z):
    return Z

def softmax_(Z):
    return Z


def tanh_(Z):
    return Z


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



def gru_forward(C_pre, x, Wr, br, Wc, bc, Wu, bu):
    r_Z = Wr.dot([C_pre, x]) + br
    r = sigmoid(r_Z)

    C_hat_Z = Wc.dot([r * C_pre, x] + bc)
    C_hat = tanh(C_hat_Z)

    u_Z = Wu.dot([C_pre, x] + br)
    u = sigmoid(u_Z)

    C = u*C_hat + (1-u)*C_pre
    y = softmax(C)

    return r_Z, r, C_hat_Z, C_hat, u_Z, u, C, y

def gru_backward(dy, dC, C_pre, x, Wr, Wc, Wu, r_Z, r, C_hat_Z, C_hat, u_Z, u, C, y):
    dC_Z = dy * softmax_(C, y) + dC
    du = (C_hat - C_pre) * dC_Z
    dC_hat = u * dC_Z
    dC_pre = (1-u) * dC_Z

    du_Z = sigmoid_(u_Z) * du
    dWu = du_Z.dot([C_pre, x])
    dbu = du_Z
    dC_pre += Wu.T.dot(du_Z)[100]

    dC_hat_Z = tanh_(C_hat_Z) * dC_hat
    dWc = dC_hat_Z.dot([r * C_pre, x].T)
    dbc = dC_hat_Z
    dmlti = Wc.T.dot(dC_hat_Z)[100]
    dr = dmlti * C_pre
    dC_pre += dmlti * r

    dr_Z = sigmoid_(r_Z) * dr
    dWr = dr_Z.dot([C_pre, x].T)
    dbr = dr_Z
    dC_pre += Wr.T.dot(dr_Z)[100]

    return dC_pre, dWu, dbu, dWc, dbc, dWr, dbr



def get_balance_corpus():
    pd_all = pd.read_csv("./../data/ChnSentiCrop_htl/ChnSentiCorp_htl_all.csv")
    corpus_pos = pd_all[pd_all.label == 1]
    corpus_neg = pd_all[pd_all.label == 0]
    sample_size = np.min(corpus_pos.shape[0], corpus_neg.shape[0])
    pd_corpus_balance = pd.concat([corpus_pos.sample(sample_size, replace=corpus_pos.shape[0] < sample_size), \
                                   corpus_neg.sample(sample_size, replace=corpus_neg.shape[0] < sample_size)])

    print('评论数目（总体）：%d' % pd_corpus_balance.shape[0])
    print('评论数目（正向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 1].shape[0])
    print('评论数目（负向）：%d' % pd_corpus_balance[pd_corpus_balance.label == 0].shape[0])

    return pd_corpus_balance

def train(itr, a):
    a = get_balance_corpus()
    a.sample(10)


if __name__ == '__main__':
    train(100, 0.01)


