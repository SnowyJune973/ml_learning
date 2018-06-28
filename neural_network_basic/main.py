import re
import numpy as np
import math
import time

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def FP(X, W):
    # X is m * (n(0)+1)
    # W[] is (n(0)+1) * (n(1))
    # X*W0[] is m * (n(1))
    dep = len(W)
    for i in range(dep):
        x[i+1] = sigmoid(X * W[i])
        m = X.shape[0]
        if i != dep-1:
            X = np.hstack((np.matrix(np.ones((m,1))), X))
        y = X
    return y

def BPwithFP(X, Y0, W, b):
    dep = len(W)
    delta_x = [0] * dep
    theta_x = [0] * dep
    x = [0] * dep
    grad = []
    grad_b = []
    #print(W[0])
    #print('********')
    #print(W[1])
    #print('********')
    #print(W[2])
    for i in range(dep):
        grad.append(np.matrix(np.zeros(W[i].shape)))
        grad_b.append(np.matrix(np.zeros(b[i].shape)))

    m = X.shape[0]
    J = 0
    #first FP
    # a NN with two hidden layer.
    for i in range(0,m):
        x[0] = X[i,:].T
        x[1] = sigmoid(W[0] * x[0] + b[0])
        x[2] = sigmoid(W[1] * x[1] + b[1])
        y = sigmoid(W[2] * x[2] + b[2])
        y0 = Y0[i,:].T
        J = J - np.multiply(y0, np.log(y))
        J = J - np.multiply(1.0-y0, np.log(1.0-y))

        delta_y = y - y0
        delta_x[2] = np.multiply(W[2].T * delta_y, np.multiply(x[2], 1.0-x[2]))
#        print(delta_x[2]
#       print(theta_x[2])
#        print('')
        delta_x[1] = np.multiply(W[1].T * delta_x[2], np.multiply(x[1], 1.0-x[1]))
#        print(delta_x[1])
#        print(theta_x[1])
        
        grad[0] = grad[0] + delta_x[1] * x[0].T
        grad[1] = grad[1] + delta_x[2] * x[1].T
        grad[2] = grad[2] + delta_y * x[2].T

        grad_b[0] = delta_x[1]
        grad_b[1] = delta_x[2]
        grad_b[2] = delta_y

    for i in range(3):
        grad[i] = grad[i] / m
        grad_b[i] = grad_b[i] / m

    J = np.sum(J) / m
    return J, grad, grad_b

def GradDownOneStep(W, b, grad, grad_b, alpha):
    dep = len(W)
    for i in range(dep):
        W[i] = W[i] - alpha * grad[i]
        b[i] = b[i] - alpha * grad_b[i]
    return W, b

def Read():
    with open('train.dt','r') as fin:
        Lst0 = fin.readlines()
    m = len(Lst0)
    raw_X = []
    raw_Y0 = []
    for i in range(m):
        Str0 = Lst0[i]
        Lst1 = re.findall('-{0,1}([0-9]*\.[0-9]+|[0-9]+)', Str0)
        yi = int(Lst1[0])
        Lst1f = [float(x) for x in Lst1]
        raw_Y0.append([0]*(yi-1)+[1]+[0]*(3-yi))
        raw_X.append(Lst1f[1:])
    X_unorganized = np.matrix(raw_X)
    Y0 = np.matrix(raw_Y0)
    aveXp = np.average(X_unorganized, 0)
    minXp = np.min(X_unorganized, 0)
    maxXp = np.max(X_unorganized, 0)
    aveX = np.tile(aveXp, (m,1))
    minX = np.tile(minXp, (m,1))
    maxX = np.tile(maxXp, (m,1))
    eps = 1e-6 
    X = np.multiply(1.0 / (maxX - minX + eps), (X_unorganized-aveX))


    with open('test.dt','r') as fin:
        Lst0 = fin.readlines()
    m = len(Lst0)
    raw_X = [] 
    raw_Y0 = []
    for i in range(m):
        Str0 = Lst0[i]
        Lst1 = re.findall('-{0,1}([0-9]*\.[0-9]+|[0-9]+)', Str0)
        yi = int(Lst1[0])
        Lst1f = [float(x) for x in Lst1]
        raw_Y0.append([0]*(yi-1)+[1]+[0]*(3-yi))
        raw_X.append(Lst1f[1:])
    Xt_unorganized = np.matrix(raw_X)
    Y0t = np.matrix(raw_Y0)
    aveXt = np.tile(aveXp, (m,1))
    minXt = np.tile(minXp, (m,1))
    maxXt = np.tile(maxXp, (m,1))
    eps = 1e-6 
    Xt = np.multiply(1.0 / (maxXt - minXt + eps), (Xt_unorganized-aveXt))
#    print(X)
    return X, Y0, Xt, Y0t
        
def Check(X, Y0, W, b):
    m = X.shape[0]
    X1 = sigmoid(X * W[0].T + np.tile(b[0].T, ((m,1))))
    X2 = sigmoid(X1 * W[1].T + np.tile(b[1].T, ((m,1))))
    Y = sigmoid(X2 * W[2].T + np.tile(b[2].T, ((m,1))))

    mxPosY = np.argmax(Y, 1) + 1
    mxPosY0 = np.argmax(Y0, 1) + 1
    diffCnt = 0
    for i in range(m):
        if mxPosY[i] != mxPosY0[i]:
            diffCnt += 1

    print('Fault Rate: ', diffCnt * 100.0 / m)

def Debug():
    X = np.matrix([[0,0],[0,1],[1,0],[1,1]])
    X = np.hstack((np.ones((X.shape[0],1)), X))
    Y0 = np.matrix([[0],[1],[1],[0]])
    W = []
    W.append(np.matrix(np.random.random((4,3))))
    W.append(np.matrix(np.random.random((2,5))))
    W.append(np.matrix(np.ones((1,3))))
    for t in range(1000):
        J, grads = BPwithFP(X, Y0, W)
        W = GradDownOneStep(W, grads, 0.5)
    print(J)
    print(grads)
    print(W)

np.set_printoptions(precision=3, suppress=True)
X, Y0, Xt, Y0t = Read()
np.random.seed(int(time.time()))

W = []
W.append(np.matrix(np.random.random((8,13))))
W.append(np.matrix(np.random.random((5,8))))
W.append(np.matrix(np.random.random((3,5))))
b = []
b.append(np.matrix(np.random.random((8,1))))
b.append(np.matrix(np.random.random((5,1))))
b.append(np.matrix(np.random.random((3,1))))

for t in range(100):
    J, grad, grad_b = BPwithFP(X, Y0, W, b)
    W, b = GradDownOneStep(W, b, grad, grad_b, 5)
#   print(J)

Check(Xt, Y0t, W, b)
print(J)
# print(grads)
