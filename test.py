import os
import sys
import numpy as np
import gzip
from PIL import Image
import scipy.io as spio



def load_dataset_variation(threat_type, variation_type, dataset_name, mode, root_path='data'):
   
    x_train, y_train, x_test, y_test = None, None, None, None
    if mode == 'train':
        fixed_path = root_path+sep+'training_set'+sep+threat_type+sep+variation_type+sep
        if dataset_name != None:
            fixed_path = root_path+sep+'training_set'+sep+threat_type+sep+dataset_name+sep+variation_type+sep

        print('loading data from', fixed_path)
        train_images = fixed_path+'train-images-npy.gz'
        train_labels = fixed_path+'train-labels-npy.gz'

        f = gzip.GzipFile(train_images, "r")
        x_train = np.load(f)
        
        f = gzip.GzipFile(train_labels, "r")
        y_train = np.load(f)

    elif mode == 'test':
        fixed_path = root_path+sep+'benchmark_dataset'+sep+threat_type+sep+variation_type+sep
        if dataset_name != None:
            fixed_path = root_path+sep+'benchmark_dataset'+sep+threat_type+sep+dataset_name+sep+variation_type+sep
        
        print('loading data from', fixed_path)
        test_images = fixed_path+'test-images-npy.gz'
        test_labels = fixed_path+'test-labels-npy.gz'

        f = gzip.GzipFile(test_images, "r")
        x_test = np.load(f)

        f = gzip.GzipFile(test_labels, "r")
        y_test = np.load(f)

    return (x_train, y_train), (x_test, y_test)


def main():

	threat_type = 'distributional_shift'
	variation = 'snow'
	severity = 1
	compl_path = '{}_severity_{}'.format(variation_severity)

	dataset_name = 'cifar10'
	mode = 'test'
	
	(x_train, y_train), (x_test, y_test) = load_dataset_variation(
		threat_type, compl_path, dataset_name, mode, root_path='../PRDC_2021_Data_profile_module/data')

	print(np.shape(x_test, y_test))

if __name__ == '__main__':
	main()