import numpy as np
import csv
from sklearn.neural_network import BernoulliRBM
from rbm import error, percent_error

def make_rbm(data, components=50, n_iter=10):
    model = BernoulliRBM(n_components=components, learning_rate=0.1, batch_size=10, n_iter=n_iter, 
    	random_state=None, verbose=0)
    model = model.fit(data)
    print("error: {}".format(percent_error(model.gibbs(data), data)))
    return model

def make_dbn(data, components=[150,50], iterations=[10,10]):
    result = []
    for size, n_iter in zip(components, iterations):
        rbm = make_rbm(data, components=size, n_iter=n_iter)
        result.append(rbm)
        #data = rbm._sample_hiddens(data, rbm.random_state_)
        #print(data.shape)
        data = np.round(rbm._mean_hiddens(data)).astype(int)
        print(data.shape)
    return result

def run_together(data, layers):
    for layer in layers:
        data = layer._sample_hiddens(data, layer.random_state_)
    for layer in reversed(layers):
        data = layer._sample_visibles(data, layer.random_state_)
    return data

if __name__ == "__main__":
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
    
    dbn = make_dbn(train_data, [512, 50], [1, 500])
    
    print("sample / sample error: {}".format(percent_error(dbn[0]._sample_hiddens(train_data, dbn[0].random_state_), dbn[0]._sample_hiddens(train_data, dbn[0].random_state_))))
    print("rounded mean / sample error: {}".format(percent_error(np.round(dbn[0]._mean_hiddens(train_data)).astype(int), dbn[0]._sample_hiddens(train_data, dbn[0].random_state_))))
    
    print("train error: {}".format(percent_error(run_together(train_data, dbn), train_data)))
    
    print("first layer total error: {}".format(percent_error(dbn[0].gibbs(test_data), test_data)))
    print("total error: {}".format(percent_error(run_together(test_data, dbn), test_data)))
