import numpy as np
import csv
import plot
from sklearn.neural_network import BernoulliRBM

def error(a, b):
	return (a != b).sum()

def percent_error(a, b):
	return sum(error(a[i], b[i]) for i in range(len(a)))/float(len(a)*len(a[0]))

def gen_even_slices(n, n_packs, n_samples=None):
    start = 0
    if n_packs < 1:
        raise ValueError("gen_even_slices got n_packs=%s, must be >=1"
                         % n_packs)
    for pack_num in range(n_packs):
        this_n = n // n_packs
        if pack_num < n % n_packs:
            this_n += 1
        if this_n > 0:
            end = start + this_n
            if n_samples is not None:
                end = min(n_samples, end)
            yield slice(start, end, None)
            start = end

def reformat_data(data):
	return data.reshape((28, 28))

train_data = []
test_data = []

with open('binMNIST_data\\bindigit_trn.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		train_data.append(np.array([int(_) for _ in row]))

with open('binMNIST_data\\bindigit_tst.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		test_data.append(np.array([int(_) for _ in row]))



batch_size=10
n_samples = np.array(train_data).shape[0]
n_batches = int(np.ceil(float(n_samples) / batch_size))
batch_slices = list(gen_even_slices(n_batches * batch_size, n_batches, n_samples))

nodes = [50, 75, 100, 150]

for item in nodes:
	errors = []
	model = BernoulliRBM(n_components=item, learning_rate=0.1, batch_size=10, n_iter=1, 
	random_state=None, verbose=1)
	for _ in range(20):
		for batch_slice in batch_slices:
			model.partial_fit(train_data[batch_slice])
		errors.append(percent_error(model.gibbs(test_data), test_data))
	plot.plot_points(errors)
	plot.plot_heatmap(reformat_data(test_data[0]))
	plot.plot_heatmap(reformat_data(model.gibbs(test_data)[0]))
	
	if item == 50 or item == 100:
		plot.plot_heatmap(model.__dict__['components_'].reshape(item,784))


