from numpy import *
import re
import math

set_printoptions(formatter={'float': '{: 0.2f}'.format},suppress=True)
def sigmoid(x):
	return 1.0/(1.0+exp(-x))

def Cost(inX, iny, param):
# param: including x[0] to x[n]. It's (n+1) * 1.
# inX is a matrix giving attributes of data. It's m * (n+1), inX[:,0] = [1 1 1 1 ... 1].
# iny is the expected output of data. It's m * 1.
	m = inX.shape[0]
	n = inX.shape[1]-1
	P1 = sigmoid(inX*param)
	print("P1 = ", P1)
	P0 = 1 - P1
	iny_1m = 1-iny
	J = -sum(array(iny)*array(log(P1))+array(iny_1m)*array(log(P0)))/m
	grad = -inX.T*(iny-P1)/m
	return J, grad

def GradDown(param, grad, alpha):
#	print("Param Old = ", param)
	param = param - alpha*grad
#	print("Param New = ", param)
	return param

def Train():
	print(kkk)
	with open('train.dt','r') as fin:
		datas = fin.readlines()
	raw_X = []
	raw_y = []
	for aStr in datas:
		raw_aLine = re.findall(r'-{0,1}[0-9]*\.[0-9]+|[0-9]+', aStr)
		raw_aLine = [float(x) for x in raw_aLine]
		raw_y.append([raw_aLine[-1]])
		del raw_aLine[-1]
		raw_X.append(raw_aLine)
	
	X = matrix(raw_X)
	y = matrix(raw_y)
	X = hstack((ones((X.shape[0],1)), X))
# print(X)
# print(y)
# print(Cost(X,y,ones((X.shape[0],1))))

	alpha = 0.1
	w = zeros((X.shape[1],1))
	for t in range(1):
		J, grad = Cost(X, y, w)
		w = GradDown(w, grad, alpha)
#	print("*********W = ", w)
		print("At Round ", t, " We get J = ", J)
		print("We get Grad = ", grad)

	return w
# print(w)

def Test_And_Judge(param):
	with open('test.dt','r') as fin:
		datas = fin.readlines()
	raw_X = []
	raw_y = []
	for aStr in datas:
		raw_aLine = re.findall(r'-{0,1}[0-9]*\.[0-9]+|[0-9]+', aStr)
		raw_aLine = [float(x) for x in raw_aLine]
		raw_y.append([raw_aLine[-1]])
		del raw_aLine[-1]
		raw_X.append(raw_aLine)
	
	X = matrix(raw_X)
	y = matrix(raw_y)
	X = hstack((ones((X.shape[0],1)), X))
	
	P1 = sigmoid(X * param)
	out_y = zeros(P1.shape)
	for i in range(P1.shape[0]):
		for j in range(P1.shape[1]):
			if(P1[i,j] == 1):
				out_y[i,j] = 1

	delta = out_y - y
	print("False Rate = ", sum(multiply(delta,delta))/P1.size)
	return Cost(X, y, param)

kkk = 3
w = Train()
print(w)
print("Cost on test dataset", Test_And_Judge(w))

