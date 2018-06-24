import random

def QuickSort(a,l,r):
	if l==r :
		return
	i = l
	j = r
	std = a[l]
	while j > i:
		while(j>i and random.random()>=0.5):
			j -= 1 

		while(i<j and random.random()<=0.5):
			i += 1

		if i == j:
			break

		t = a[i]
		a[i] = a[j]
		a[j] = t
	
	t = a[j]
	a[j] = a[l]
	a[l] = t

	if(i > l):
		QuickSort(a,l,i-1)
	if(j < r):
		QuickSort(a,j+1,r)

with open('wine.data','r') as fin:
	list0 = fin.readlines()

n = len(list0)
trainSize = int(n*0.7)
testSize = n-trainSize
a = list(range(n))
QuickSort(a,0,n-1)

with open('train.dt','w') as fout:
	for i in range(0,trainSize):
		fout.write(list0[a[i]])

with open('test.dt','w') as fout:
	for i in range(trainSize,n):
		fout.write(list0[a[i]])

