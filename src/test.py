from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import hardness_callback as H
import noise as N
import numpy as np



def get_conf(arr_conf):
	arr = []
	for conf in arr_conf:
		ind = np.argmax(conf)
		arr.append(conf[ind])
	return arr


def main():
	batch_size = 128
	num_classes = 10
	epochs = 12
	number_of_partitions = 4

	# input image dimensions
	img_rows, img_cols = 28, 28

	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	# Limit the data to n samples
	# num_samples = 100
	# x_train = x_train[:num_samples]
	# y_train = y_train[:num_samples]
	# x_test = x_test[:num_samples]
	# y_test = y_test[:num_samples]

	if K.image_data_format() == 'channels_first':
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train_matrix = keras.utils.np_utils.to_categorical(y_train, num_classes)
	y_test_matrix = keras.utils.np_utils.to_categorical(y_test, num_classes)

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=tf.keras.optimizers.Adam(),
	              metrics=['accuracy'])

	checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
	    save_best_only=True, mode='auto')#, save_freq="epoch")

	hardness_obj = H.HardnessCallback(x_train, y_train_matrix, epochs, number_of_partitions)

	model.fit(x_train, y_train_matrix,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(x_test, y_test_matrix),
	          callbacks=[checkpoint, hardness_obj]
	          )

	#score = model.evaluate(x_test, y_test_matrix, verbose=0)
	#print('Test loss:', score[0])
	#print('Test accuracy:', score[1])

	noise_object = N.Noise(x_train, y_train)
	noise_scores = noise_object.calculate_noise()

	a = hardness_obj.arr_hardness_degree
	aa = hardness_obj.arr_ind_learntImg_epoch
	b = noise_scores
	arr_conf = get_conf(model.predict(x_train))

	for j in range(0,10):
		c = []#hardness
		cc = []#num epochs of correctly learn img
		d = []#noise
		e = []#is_class_learned
		f = []#confidence
		
		for i in range(len(y_train)):
			
			if y_train[i]==j:
				c.append(a[i])
				cc.append(aa[i])
				d.append(b[i])
				is_learnt = False if a[i]==99 else True
				e.append(is_learnt)
				f.append(arr_conf[i])

		np.save('results\\H\\hardness_{}.txt'.format(j), c)
		np.save('results\\H\\num_epoch_hardness_{}.txt'.format(j), cc)
		np.save('results\\N\\noise_{}.txt'.format(j), d)
		np.save('results\\I\\is_learned_{}.txt'.format(j), e)
		np.save('results\\C\\confidences_{}.txt'.format(j), f)
