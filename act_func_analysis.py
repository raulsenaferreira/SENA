# external libs
import os
from scipy import spatial
import pickle
import numpy as np

# internal libs
from dataset import Dataset
from feature_extractor import FeatureExtractor


def similarity_act_func(id_dataset, features, id_y_train, pred_train, root_path):
    for cls in range(10):
        filename_c = os.path.join(root_path, "{}_class_{}_correct.sav".format(id_dataset, cls))
        filename_i = os.path.join(root_path, "{}_class_{}_incorrect.sav".format(id_dataset, cls))
        scores_correct = []
        scores_incorrect = []

        # all labels from class c
        ind_y_c = np.where(id_y_train == cls)[0]

        # all pred as c
        ind_ML_c = np.where(pred_train == cls)[0]

        # features from correct pred
        ind = set(ind_y_c).intersection(ind_ML_c)
        f_c_correct = features[list(ind)]

        # features from incorrect pred
        ind = set(ind_y_c).symmetric_difference(ind_ML_c)
        f_c_incorrect = features[list(ind)]

        # how similar are act functions between themselves?
        for i in f_c_incorrect:
            for c in f_c_correct:
                cosine_similarity = 1 - spatial.distance.cosine(c, i)
                scores_incorrect.append(cosine_similarity)

        for i in range(1, len(f_c_correct)):
            cosine_similarity = 1 - spatial.distance.cosine(f_c_correct[i - 1], f_c_correct[i])
            scores_correct.append(cosine_similarity)

        pickle.dump(scores_incorrect, open(filename_i, 'wb'))
        pickle.dump(scores_correct, open(filename_c, 'wb'))

        print('avg sim correct/incorrect pred', np.sum(scores_incorrect) / len(scores_incorrect))
        print('avg sim between pairs de correct pred', np.sum(scores_correct) / len(scores_correct))


# model params
batch_size = 10
model = "resnet"

# monitor params
id_layer_monitored = -1
layer_relu_ids = [32]
additional_transform = None
adversarial_attack = None

# dataset params
id_dataset = "cifar10" #svhn
dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)

feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
_, features_train, _, _, pred_train, lab_train = feature_extractor.get_features(dataset_train)

# training set
print('analysis for training set')
root_path = os.path.join("Features", model)
features = features_train[id_layer_monitored]
similarity_act_func(id_dataset, features, lab_train, pred_train, root_path)
