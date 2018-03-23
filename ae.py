from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import History 
import csv
import numpy as np
import plot

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

train_data = np.array(train_data)
test_data = np.array(test_data)

nodes = [50, 75, 100, 150]

for item in nodes:
	encoding_dim = item

	# this is our input placeholder
	input_img = Input(shape=(784,))
	encoded = Dense(encoding_dim, activation='relu')(input_img)
	decoded = Dense(784, activation='sigmoid')(encoded)
	autoencoder = Model(input_img, decoded)

	autoencoder.compile(optimizer='SGD', loss='binary_crossentropy', metrics=['binary_accuracy'])

	history = History()

	autoencoder.fit(train_data, train_data,
           		epochs=20,
                batch_size=10,
                shuffle=True,
                verbose=0,
                callbacks=[history],
                validation_data=(test_data, test_data))

	errors = [1-x for x in history.history['val_binary_accuracy']]
	

	plot.plot_points(errors)
	plot.plot_heatmap(reformat_data(test_data[0]))
	plot.plot_heatmap(reformat_data(autoencoder.predict(test_data)[0]))
	if item == 50 or item == 100:
		print(plot.plot_heatmap(autoencoder.layers[1].get_weights()[0].T))