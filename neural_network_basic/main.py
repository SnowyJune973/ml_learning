import re
import numpy as np
import math

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

def BPwithFP(X, Y0, W):
    dep = len(W)
    delta_x = [0] * dep
    theta_x = [0] * dep
    x = [0] * dep
    grad = []
    for i in range(dep):
        grad.append(np.matrix(np.zeros(W[i].shape)))

    m = X.shape[0]
    J = 0
    #first FP
    # a NN with two hidden layer.
    for i in range(0,m):
        x[0] = X[i,:].T
        x[1] = sigmoid(W[0] * x[0])
        x[1] = np.vstack((np.matrix([[1]]), x[1]))
        x[2] = sigmoid(W[1] * x[1])
        x[2] = np.vstack((np.matrix([[1]]), x[2]))
        y = sigmoid(W[2] * x[2])
        y0 = Y0[i,:].T
        J = J - np.multiply(y0, np.log(y))
        J = J - np.multiply(1.0-y0, np.log(1.0-y))
        theta_y = y - y0
        delta_x[2] = (W[2].T * theta_y)[1:]
#        print(delta_x[2])
        theta_x[2] = np.multiply(np.multiply(delta_x[2], x[2][1:]), (1.0 - x[2][1:]))
#       print(theta_x[2])
#        print('')
        delta_x[1] = (W[1].T * theta_x[2])[1:]
#        print(delta_x[1])
        theta_x[1] = np.multiply(np.multiply(delta_x[1], x[1][1:]), (1.0 - x[1][1:]))
#        print(theta_x[1])
        
        grad[0] = grad[0] + theta_x[1] * x[0].T
        grad[1] = grad[1] + theta_x[2] * x[1].T
        grad[2] = grad[2] + theta_y * x[2].T

    grad[0] = grad[0] / m
    grad[1] = grad[1] / m
    grad[2] = grad[2] / m
    J = np.sum(J) / m
    return J, grad

def GradDownOneStep(W, grads, alpha):
    dep = len(W)
    for i in range(dep):
        W[i] = W[i] - alpha * grads[i]
    return W

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
    aveX = np.tile(np.average(X_unorganized, 0), (m,1))
    minX = np.tile(np.min(X_unorganized, 0), (m,1))
    maxX = np.tile(np.max(X_unorganized, 0), (m,1))
    eps = 1e-6
    
    X = np.multiply(1.0 / (maxX - minX + eps), (X_unorganized-aveX))
    X = np.hstack((np.matrix(np.ones((m,1))), X))
#    print(X)
    return X, Y0
        
def Check(X, Y0, W):
    m = X.shape[0]
    X1 = np.hstack((np.matrix(np.ones((m,1))), sigmoid(X * W[0].T)))
    X2 = np.hstack((np.matrix(np.ones((m,1))), sigmoid(X1 * W[1].T)))
    Y = sigmoid(X2 * W[2].T)
    print(Y)

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
    W.append(np.matrix(np.ones((4,3))))
    W.append(np.matrix(np.ones((2,5))))
    W.append(np.matrix(np.ones((1,3))))
    for t in range(1000):
        J, grads = BPwithFP(X, Y0, W)
        W = GradDownOneStep(W, grads, 0.5)
    print(J)
    print(grads)
    print(W)

np.set_printoptions(precision=3, suppress=True)
X, Y0 = Read()
W = []
W.append(np.matrix(np.ones((8,14))))
W.append(np.matrix(np.ones((5,9))))
W.append(np.matrix(np.ones((3,6))))
for t in range(1200):
    J, grads = BPwithFP(X, Y0, W)
    W = GradDownOneStep(W, grads, 0.5)
#   print(J)

Check(X, Y0, W)
print(J)
# print(grads)
