import os
import numpy as np
from dataset import Dataset
from feature_extractor import FeatureExtractor
from methods import shine
import pickle
from joblib import Parallel, delayed


def get_indices_correct_incorrect_predictions(cls, labels, predictions):
    # all labels from class c
    ind_y_c = np.where(labels == cls)[0]
    # all pred as c
    ind_ML_c = np.where(predictions == cls)[0]
    # all correct pred as c (TP)
    ind_tp = set(ind_y_c).intersection(ind_ML_c)
    # all incorrectly pred from the true class c (FN)
    ind_fn = set(ind_y_c).symmetric_difference(ind_ML_c)

    return ind_tp, ind_fn


def fit_density_estimator(data):
    density_estimator_feature = shine.calculate_density(data)

    return density_estimator_feature


def fit_kde(num_classes, data_type, x_train, features, lab_train, pred_train,
            root_path, filename_format_c, filename_format_i):

    for cls in range(num_classes):
        pred_c_correct = []
        pred_c_incorrect = []
        kde_feature_c = None
        kde_feature_inc = None

        filename_c = os.path.join(root_path, data_type, filename_format_c.format(cls))
        filename_i = os.path.join(root_path, data_type, filename_format_i.format(cls))

        ind_tp, ind_fn = get_indices_correct_incorrect_predictions(cls, lab_train, pred_train)

        print("Generating KDE {} for class {}:".format(data_type, cls))
        if "features" in data_type:
            # features from correct pred
            pred_c_correct = features[list(ind_tp)]
            # features from incorrect pred
            pred_c_incorrect = features[list(ind_fn)]
        elif "shine" in data_type:
            # shine vectors from correct pred
            x_train_correct = x_train[list(ind_tp)]
            pred_c_correct = shine.create_matrix_shine(x_train_correct)
            # shine vectors from incorrect pred
            x_train_incorrect = x_train[list(ind_fn)]
            pred_c_incorrect = shine.create_matrix_shine(x_train_incorrect)

        # density estimator fitted with correct predictions
        if len(pred_c_correct) > 1:
            kde_feature_c = fit_density_estimator(pred_c_correct)
        # density estimator fitted with correct predictions
        if len(pred_c_incorrect) > 1:
            kde_feature_inc = fit_density_estimator(pred_c_incorrect)

        print("Saving KDE {} for class {}:".format(data_type, cls))
        if kde_feature_c is not None:
            pickle.dump(kde_feature_c, open(filename_c, 'wb'))
        if kde_feature_inc is not None:
            pickle.dump(kde_feature_inc, open(filename_i, 'wb'))


if __name__ == '__main__':
    n_jobs = 4
    batch_size = 10
    model = "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]
    additional_transform = None
    adversarial_attack = None

    # dataset params
    id_dataset = "cifar10"  # svhn
    num_classes = 10
    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    # training set
    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, _, _, pred_train, lab_train = feature_extractor.get_features(dataset_train)
    features = features_train[id_layer_monitored]

    root_path = os.path.join("methods", "fitted_kde", model, id_dataset)
    filename_format_c = "kde_cls_{}_tp.sav"
    filename_format_i = "kde_cls_{}_fn.sav"

    arr_data_type = ["on_shine_vectors", "on_features"]

    Parallel(n_jobs=n_jobs)(
        delayed(fit_kde)(num_classes,
                         data_type,
                         x_train,
                         features,
                         lab_train,
                         pred_train,
                         root_path,
                         filename_format_c,
                         filename_format_i)
        for data_type in arr_data_type
    )
