import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import Counter



# code based on the paper "Hardness of Samples Is All You Need: Protecting Deep Learning Models Using Hardness of Samples"
# https://arxiv.org/pdf/2106.11424.pdf
class HardnessCallback(keras.callbacks.Callback):

    def __init__(self, X, y, num_epochs, number_of_partitions, verbose=False):
        super(HardnessCallback, self).__init__()
        self.X = X
        self.y = y
        self.num_epochs = num_epochs
        self.number_of_partitions = number_of_partitions
        self.verbose = verbose
        self.len_data = len(X)


    def calculate_hardness_degree(self, epoch):
        predict = self.model.predict(self.X)

        for i in range(len(predict)):
            pred = np.argmax(predict[i])
            lbl = np.argmax(self.y[i])

            if pred == lbl:
                self.arr_ind_learntImg_epoch[i]+=1
        
        if (int(epoch)+1) % self.partition_size == 0:
            self.partition_counter += 1
            print('\nepoch:{} partition size:{}'.format((int(epoch)+1), self.partition_counter))
            for i in range(len(self.arr_ind_learntImg_epoch)):
                if self.arr_ind_learntImg_epoch[i] == self.partition_size:
                    self.arr_hardness_degree[i] = self.partition_counter  


    def calculate_hardness_degree_2(self, epoch):
        predict = self.model.predict(self.X)

        for i in range(len(predict)):
            pred = np.argmax(predict[i])
            lbl = np.argmax(self.y[i])

            if pred == lbl:
                self.arr_ind_learntImg_epoch[i]+=1
        
        if (int(epoch)+1) % self.partition_size == 0:
            self.partition_counter += 1
            print('\nepoch:{} partition size:{}'.format((int(epoch)+1), self.partition_counter))
            for i in range(len(self.arr_ind_learntImg_epoch)):
                if self.arr_ind_learntImg_epoch[i] == self.partition_size:
                    self.arr_hardness_degree[i] = self.partition_counter


    def calculate_trust_magnitude(self):
        self.hardness_magnitudes_per_rank_degree = dict(Counter(self.arr_hardness_degree))


    def on_train_begin(self, logs=None):
        # The index on the next two arrays represents the index of the train imgs. 
        # The first one stores, for each index, the number of times that the img was correctly learnt
        # The second one stores, for each index, the partition number that the img was correctly learnt

        self.arr_ind_learntImg_epoch = [0]*self.len_data

        self.arr_hardness_degree = [99]*self.len_data #99 means the instances were not learned

        # number of submodels for each partition
        self.partition_size = int(self.num_epochs/self.number_of_partitions)
        
        self.partition_counter = 0
        
        if self.verbose:
            print("Starting training...")


    def on_train_end(self, logs=None):

        print("Stoping training...")
        #calculate the percentage of correct classifications per hardness rank
        for k in sorted(self.hardness_magnitudes_per_rank_degree.keys()):
            v = self.hardness_magnitudes_per_rank_degree[k]
            print('Rank {}: {:3.2f}%'.format(k, v/self.len_data*100))


    def on_epoch_begin(self, epoch, logs=None):
        
        print("Starting epoch...")
        

    def on_epoch_end(self, epoch, logs=None):
        
        self.calculate_hardness_degree(epoch)  
        if self.verbose:
            print("\nHardness degrees: {}".format(epoch, self.arr_hardness_degree))

        self.calculate_trust_magnitude()
        if self.verbose:
            print('\ntrust_magnitude_per_rank_degree:', self.hardness_magnitudes_per_rank_degree)
