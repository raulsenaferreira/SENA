# external libs
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcc
from scipy import spatial
from statistics import NormalDist

import utils
# internal libs
from dataset import Dataset
from feature_extractor import FeatureExtractor
from methods import shine


def calculate_feature_similarity(features_1, features_2, mode="fn", sim_type="cosine"):
    scores = []
    # similarities
    if mode == "fn":
        for fn in features_1:
            for tp in features_2:
                if sim_type == "cosine":
                    cosine_similarity = 1 - spatial.distance.cosine(tp, fn)
                    scores.append(cosine_similarity)
                elif sim_type == "euclidean":
                    eucl_dist = np.linalg.norm(tp - fn)
                    scores.append(eucl_dist)
    elif mode == "tp":
        for i in range(len(features_1) - 1):
            if sim_type == "cosine":
                cosine_similarity = 1 - spatial.distance.cosine(features_1[i], features_1[i + 1])
                scores.append(cosine_similarity)
            elif sim_type == "euclidean":
                eucl_dist = np.linalg.norm(features_1[i] - features_1[i + 1])
                scores.append(eucl_dist)
    elif mode == "single":  # assuming features_1 is just one instance (incoming feature)
        if sim_type == "cosine":
            for c in features_2:
                cosine_similarity = 1 - spatial.distance.cosine(c, features_1)
                scores.append(cosine_similarity)
        elif sim_type == "euclidean":
            scores = np.linalg.norm(features_1 - features_2, ord=2, axis=1)

    return scores


def calculate_feature_similarity_thresholds(num_classes, features, y, ml_predictions):
    features = np.array(features)
    features_tp_by_class = {}
    similarities_TP_by_class = {}
    features_fn_by_class = {}
    similarities_FN_by_class = {}

    for c in range(num_classes):
        # all labels from class c
        ind_y_c = np.where(y == c)[0]

        # Correct predictions
        ind_ML_correct = np.where(ml_predictions == c)[0]
        # Incorrect predictions
        ind_incorrect_ML = set(ind_y_c).symmetric_difference(ind_ML_correct)

        # features from TP and FN
        features_tp = features[list(ind_ML_correct)]
        features_tp_by_class.update({c: features_tp})
        scores_TP = calculate_feature_similarity(features_tp, features_tp, "tp", "euclidean")
        similarities_TP_by_class.update({c: scores_TP})

        if len(ind_incorrect_ML) > 1:
            features_fn = features[list(ind_incorrect_ML)]
            scores_FN = calculate_feature_similarity(features_fn, features_tp, "fn", "euclidean")

            features_fn_by_class.update({c: features_fn})
            similarities_FN_by_class.update({c: scores_FN})
        else:
            features_fn_by_class.update({c: None})
            similarities_FN_by_class.update({c: None})

    return features_tp_by_class, features_fn_by_class, similarities_TP_by_class, similarities_FN_by_class


def save_test_results(results, approach, id_dataset, ood_type, variant, ground_truth):
    root_path = os.path.join('test', 'results')
    filename_gt = os.path.join(root_path, id_dataset, ood_type, variant, 'ground_truth.sav')
    pickle.dump(ground_truth, open(filename_gt, 'wb'))

    filename = os.path.join(root_path, id_dataset, ood_type, variant, '{}.sav'.format(approach))
    print("saving results from approach {} for {}".format(approach, ood_type))
    pickle.dump(results, open(filename, 'wb'))


def fit_density_estimator(data):
    density_estimator_feature = shine.calculate_density(data)

    return density_estimator_feature


def get_thresholded_data(data, threshold, magnitude=1):
    if data is not None:
        len_data = len(data)
        if threshold == 'new':
            std = np.std(data)
            third_quartile = np.median(data[int(len(data) / 2):])
            threshold_val = third_quartile + std * magnitude  # it's based on the overlapping region between two dists
        else:
            sorted_data = sorted(data)
            id_threshold = int(len_data * threshold)
            threshold_val = sorted_data[id_threshold]
        return threshold_val
    else:
        return None


def predict_original(incoming_feature, features_tp_by_class, similarities_TP_by_class, magnitude):
    # caculate similarities
    incoming_tp_scores = calculate_feature_similarity(incoming_feature, features_tp_by_class, "single", "euclidean")

    threshold_TP = get_thresholded_data(similarities_TP_by_class, 'new', magnitude)
    if np.mean(incoming_tp_scores) >= max(similarities_TP_by_class):  # it is probably an FN (incorrect pred)
        return True
    else:
        if np.mean(incoming_tp_scores) <= threshold_TP:
            return False
        else:
            return True


def load_results_test(approach, id_dataset, ood_type, variant):
    root_path = os.path.join('test', 'results')
    filename_gt = os.path.join(root_path, id_dataset, ood_type, variant, 'ground_truth.sav')
    ground_truth = pickle.load(open(filename_gt, 'rb'))

    filename = os.path.join(root_path, id_dataset, ood_type, variant, '{}.sav'.format(approach))
    results = pickle.load(open(filename, 'rb'))

    return ground_truth, results


def load_pdf_tp_for_test(num_cls, model):
    arr_pdf_on_features_tp = {}

    root_path = os.path.join("data_analysis", model, id_dataset)
    filename_format_c = "cls_{}_correct_correct.sav"

    for cls in range(num_cls):
        filename_feature_tp = os.path.join(root_path, "feature_pdf", filename_format_c.format(cls))
        scores_feature_pdf_tp = pickle.load(open(filename_feature_tp, 'rb'))
        arr_pdf_on_features_tp.update({cls: scores_feature_pdf_tp})

    return arr_pdf_on_features_tp


def main(model, layer_relu_ids, percentage_repr_tp, id_dataset, num_classes, ood_dataset, magnitude):
    # model params
    global scaler
    batch_size = 100

    # monitor params
    id_layer_monitored = -1

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
        dataset_train)
    x_test, features_test, logits_test, softmax_test, pred_test, lab_test = feature_extractor.get_features(
        dataset_test)

    if ood_dataset == "cifar10" or ood_dataset == "svhn" or ood_dataset == "gtsrb" or ood_dataset == "cifar100":
        dataset_ood = Dataset(ood_dataset, "train", model, batch_size=batch_size)
    else:
        dataset_ood = Dataset(ood_dataset, "test", model, batch_size=batch_size)

    x_ood, features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

    # accuracy of the ML model
    id_accuracy = accuracy_score(lab_test, pred_test)
    ood_accuracy = 0
    if id_dataset == ood_dataset:
        ood_accuracy = accuracy_score(lab_ood, pred_ood)
    print("Model accuracy")
    print("ID:  ", id_accuracy)
    print("OOD: ", ood_accuracy)

    arr_pdf_on_features_tp = load_pdf_tp_for_test(num_classes, model)
    # selecting the k most representative entries from training
    features_train = np.array(features_train[id_layer_monitored])

    features_tp_by_class = {}
    similarities_TP_by_class = {}
    features_fn_by_class = {}
    similarities_FN_by_class = {}
    pca_models_by_class = {}

    for c in range(num_classes):
        k = int(len(arr_pdf_on_features_tp[c]) / 100 * percentage_repr_tp)
        # print("getting the {} most representative true positives of the class {}".format(k, c))
        top_ind = np.argpartition(arr_pdf_on_features_tp[c], -k)[-k:]
        # all labels from class c
        ind_y_c = np.where(lab_train == c)[0]
        # Correct predictions
        ind_ML_correct = np.where(pred_train == c)[0]
        # Incorrect predictions
        ind_incorrect_ML = set(ind_y_c).symmetric_difference(ind_ML_correct)
        # features from TP and FN
        # features_tp = features_train[list(ind_ML_correct)]
        features_tp = features_train[list(ind_ML_correct)][top_ind]  # storing only k-representative fp

        # optional: transforming sparse vectors in dense ones
        if use_pca:
            from sklearn.decomposition import PCA, TruncatedSVD
            my_model = PCA(n_components=0.99, svd_solver='full')
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaler.fit(features_tp)
            features_tp_rescaled = scaler.transform(features_tp)
            my_model.fit(features_tp_rescaled)
            pca_models_by_class.update({c: my_model})

            dense_features_tp = my_model.transform(features_tp_rescaled)
            features_tp_by_class.update({c: dense_features_tp})

            scores_TP = calculate_feature_similarity(dense_features_tp, dense_features_tp, "tp", "euclidean")
            similarities_TP_by_class.update({c: scores_TP})

        else:
            features_tp_by_class.update({c: features_tp})
            scores_TP = calculate_feature_similarity(features_tp, features_tp, "tp", "euclidean")
            similarities_TP_by_class.update({c: scores_TP})

        if len(ind_incorrect_ML) > 1:
            features_fn = features_train[list(ind_incorrect_ML)]
            if use_pca:
                scaler.transform(features_fn)
                dense_features_fn = pca_models_by_class[c].transform(features_fn)
                scores_FN = calculate_feature_similarity(dense_features_fn, features_tp_by_class[c], "fn", "euclidean")
                similarities_FN_by_class.update({c: scores_FN})
                features_fn_by_class.update({c: dense_features_fn})

            else:
                scores_FN = calculate_feature_similarity(features_fn, features_tp, "fn", "euclidean")
                similarities_FN_by_class.update({c: scores_FN})
                features_fn_by_class.update({c: features_fn})
        else:
            features_fn_by_class.update({c: None})
            similarities_FN_by_class.update({c: None})

        # optional, if you want to plot the distributions of TP and FN for an ID dataset
        utils.plot_dist_fn_tp(similarities_TP_by_class[c], similarities_FN_by_class[c])

    # testing approaches
    m_true = []
    monitor_results = []

    # test data (ID)
    for x, pred, feature, softmax, logits, label in tqdm(
            zip(x_test, pred_test, features_test[id_layer_monitored],
                softmax_test, logits_test, lab_test)):

        if pred == label:  # monitor does not need to activate
            m_true.append(False)
        else:  # monitor should activate
            m_true.append(True)

        data_source_1 = feature
        if use_pca:
            data_source_1_scaled = scaler.transform([feature])
            data_source_1 = pca_models_by_class[pred].transform(data_source_1_scaled)

        result = predict_original(data_source_1, features_tp_by_class[pred],
                                  similarities_TP_by_class[pred], magnitude)
        monitor_results.append(result)

    print("ID MCC: {}, classification report: {}".format(mcc(y_true=m_true, y_pred=monitor_results),
                                                         confusion_matrix(m_true, monitor_results)))
    # novelty data
    m_ood = []
    monitor_ood_results = []
    for x, feature, softmax, logits, pred, label in tqdm(
            zip(x_ood, features_ood[id_layer_monitored], softmax_ood,
                logits_ood, pred_ood, lab_ood)):
        m_ood.append(True)
        m_true.append(True)

        data_source_1 = feature
        if use_pca:
            data_source_1_scaled = scaler.transform([feature])
            data_source_1 = pca_models_by_class[pred].transform(data_source_1_scaled)

        result = predict_original(data_source_1, features_tp_by_class[pred],
                                  similarities_TP_by_class[pred], magnitude)
        monitor_results.append(result)
        monitor_ood_results.append(result)

    print("OOD MCC: {} and confusion matrix: {}".format(mcc(y_true=m_ood, y_pred=monitor_ood_results),
                                                        confusion_matrix(m_ood, monitor_ood_results)))
    return monitor_results, m_true


if __name__ == '__main__':
    # for each new ID dataset, execute data analysis script for creating a cached data (for speed experiment reasons)
    model = "resnet"  # "resnet" "cnn"
    layer_relu_ids = [32]  # [32] [4]
    # dataset params
    ood_type = "novelty"
    # additional_transforms = ["blur", "brightness", "pixelization", "shuffle_pixels"]
    id_dataset = "cifar10"  # "cifar10", "svhn" "mnist"#
    arr_ood_dataset = ["svhn", "gtsrb", "lsun", "tiny_imagenet", "cifar100", "fractal"]
    # id_dataset = 'mnist'
    # arr_ood_dataset = ["fashion_mnist", "emnist", "asl_mnist", "simpsons_mnist"]
    magnitude = 1  # 2 for CIFAR; 1 for SVHN and mnist
    num_classes = 10
    arr_results = []
    approach_name = 'sena'
    percentage_repr_tp = 10
    use_pca = False

    for ood_dataset in arr_ood_dataset:
        monitor_results, m_true = main(model, layer_relu_ids, percentage_repr_tp,
                                       id_dataset, num_classes, ood_dataset, magnitude)

        # saving results
        save_test_results(monitor_results, approach_name, id_dataset, ood_type, ood_dataset, m_true)
        # loading results
        ground_truth, results = load_results_test(approach_name, id_dataset, ood_type, ood_dataset)

        utils.print_results_test_single(id_dataset, ood_type, approach_name, results, ground_truth, ood_dataset)
