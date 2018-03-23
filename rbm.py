import numpy as np
import csv
from sklearn.neural_network import BernoulliRBM

def error(a, b):
	return (a != b).sum()

def percent_error(a, b):
	return sum(error(a[i], b[i]) for i in range(len(a)))/float(len(a)*len(a[0]))

train_data = []

with open('binMNIST_data\\bindigit_trn.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		train_data.append(np.array([int(_) for _ in row]))

model = BernoulliRBM(n_components=50, learning_rate=0.1, batch_size=10, n_iter=1, 
	random_state=None, verbose=1)

model = model.fit(train_data)
print(percent_error(model.gibbs(train_data), train_data))

