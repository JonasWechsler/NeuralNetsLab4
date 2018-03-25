import numpy as np
import csv
from sklearn.neural_network import BernoulliRBM, MLPClassifier
import plot
from rbm import error, percent_error, reformat_data, gen_even_slices

def make_rbm(data, components=50, n_iter=20):
    model = BernoulliRBM(n_components=components, learning_rate=0.1, batch_size=10, n_iter=n_iter, 
    	random_state=None, verbose=0)
    model = model.fit(data)
    print("error: {}".format(percent_error(model.gibbs(data), data)))
    return model

def make_dbn(data, components=[150,50], iterations=None):
    if iterations == None:
        iterations = [1]*len(components)
    result = []
    for size, n_iter in zip(components, iterations):
        rbm = make_rbm(data, components=size, n_iter=n_iter)
        result.append(rbm)
        data = np.round(rbm._mean_hiddens(data)).astype(int)
    return result

def run_together(data, layers):
    for layer in layers:
        data = layer._sample_hiddens(data, layer.random_state_)
    for layer in reversed(layers):
        data = layer._sample_visibles(data, layer.random_state_)
    return data

def read_to_array(inf):
    data = []
    with open(inf) as f:
    	reader = csv.reader(f)
    	for row in reader:
    		data.append(np.array([int(_) for _ in row]))
    return np.array(data)


if __name__ == "__main__":
    train_data = read_to_array('binMNIST_data\\bindigit_trn.csv')
    test_data = read_to_array('binMNIST_data\\bindigit_tst.csv')
    train_class = read_to_array('binMNIST_data\\targetdigit_trn.csv')
    test_class = read_to_array('binMNIST_data\\targetdigit_tst.csv')
    print("train data: {}, test data: {}".format(len(train_data), len(test_data)))
    
    dbn = make_dbn(train_data, [150, 100], [10, 100, 500])
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(784, 512), random_state=1)
    mlp.fit(train_data, train_class)
    
    print("sample / sample error: {}".format(percent_error(dbn[0]._sample_hiddens(train_data, dbn[0].random_state_), dbn[0]._sample_hiddens(train_data, dbn[0].random_state_))))
    print("rounded mean / sample error: {}".format(percent_error(np.round(dbn[0]._mean_hiddens(train_data)).astype(int), dbn[0]._sample_hiddens(train_data, dbn[0].random_state_))))
    
    #for i in [27, 4, 18, 1, 3, 2, 15, 10, 7, 6]:
    #    #plot.plot_heatmap(reformat_data(test_data[i-1]))
    #    #plot.plot_heatmap(reformat_data(dbn[0].gibbs(test_data[i-1])))
    #    plot.plot_heatmap(reformat_data(run_together(test_data[i-1], dbn)))
    
    result = run_together(test_data, dbn)
    rbm_result = dbn[0].gibbs(test_data)
    print("first layer total error: {}".format(percent_error(rbm_result, test_data)))
    print("total error: {}".format(percent_error(result, test_data)))
    print(sum([1 if a == b else 0 for a, b in zip(mlp.predict(result), test_class)])/len(result))
    print(sum([1 if a == b else 0 for a, b in zip(mlp.predict(rbm_result), test_class)])/len(rbm_result))
