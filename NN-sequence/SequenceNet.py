''''
one hot encoding: https://blog.csdn.net/dulingtingzi/article/details/51374487
负采样的核心思想是，如果将自然语言看作是一串单词的随机组合，那么它的出现概率是很小的。于是，如果我们将拼凑的单词随机组合（负采样）起来将会以很大的概率不会出现在当前文章中。于是，我们很显然应该至少让我们的模型在这些负采样出来的单词组合上面出现概率应该尽可能地小，同时要让真正出现在文中的单词组合出现概率大
'''

if __name__ == '__main__':
    print("龙".encode().decode('utf-8'))

def sigmoid(Z):
    return Z

def sigmoid_(Z):
    return Z

def tanh(Z):
    return Z

def tanh_(Z):
    return Z

def softmax(Z):
    return Z

def softmax_(Z):
    return Z

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



def train( )