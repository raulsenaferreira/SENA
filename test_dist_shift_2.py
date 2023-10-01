# external libs
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcc
from scipy import spatial

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
    '''
    bp_incoming_X = plt.boxplot(scores, showmeans=True)
    # getting thresholds based on quartil 1 for TP and FN
    min_acc_threshold = list(set(bp_incoming_X['boxes'][0].get_ydata()))[0]
    # min_acc_threshold = list(set(bp_incoming_X['means'][0].get_ydata()))
    '''
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


def save_test_results(arr_results, arr_approach_names, id_dataset, ood_type, variant, ground_truth):
    root_path = os.path.join('test', 'results')
    filename_gt = os.path.join(root_path, id_dataset, ood_type, variant, 'ground_truth.sav')
    pickle.dump(ground_truth, open(filename_gt, 'wb'))

    for results, approach in zip(arr_results, arr_approach_names):
        filename = os.path.join(root_path, id_dataset, ood_type, variant, '{}.sav'.format(approach))
        print("saving results from approach {} for {}".format(approach, ood_type))
        pickle.dump(results, open(filename, 'wb'))


def fit_density_estimator(data):
    density_estimator_feature = shine.calculate_density(data)

    return density_estimator_feature


def generate_kde_intersection_tp_fn(pdfs_fn, pdfs_tp, x_train):
    density_estimator_tp_fn = None
    pdfs_tp_fn = None
    indices_pdf_intersection_tp_fn = []
    min_pdf_tp = min(pdfs_tp)

    for i in range(len(pdfs_fn)):
        if pdfs_fn[i] >= min_pdf_tp:
            indices_pdf_intersection_tp_fn.append(i)

    if len(indices_pdf_intersection_tp_fn) > 1:
        selected_imgs = x_train[indices_pdf_intersection_tp_fn]
        matrix_shine = shine.create_matrix_shine(selected_imgs)
        density_estimator_tp_fn = fit_density_estimator(matrix_shine)
        pdfs_tp_fn = density_estimator_tp_fn.score_samples(matrix_shine)

    return density_estimator_tp_fn, pdfs_tp_fn


def predict_with_tp(data, pdfs_tp, density_estimator_tp, data_type, threshold):
    incoming_pdf = None
    threshold_pdf_tp = None

    if threshold == 'min':
        threshold_pdf_tp = min(pdfs_tp)
    elif threshold == 'avg':
        threshold_pdf_tp = np.mean(pdfs_tp)

    if "shine" in data_type:
        matrix_shine = shine.create_matrix_shine(data)
        incoming_pdf = np.exp(density_estimator_tp.score_samples(matrix_shine))[0]
    elif "feature" in data_type:
        incoming_pdf = np.exp(density_estimator_tp.score_samples(data))[0]

    if incoming_pdf >= threshold_pdf_tp:
        return False  # no need to intervene, it's probably an TP
    else:
        return True  # it's probably an FN


def predict_with_tp_fn_hybrid(data1, data2, pdfs_tp_source_1, density_estimator_tp_source_1,
                              pdfs_fn_source_2, density_estimator_fn_source_2, data_type, threshold):
    incoming_pdf = None
    new_pdf = None
    threshold_pdf_tp = None
    threshold_pdf_fn = None

    if threshold == 'min':
        threshold_pdf_tp = min(pdfs_tp_source_1)
        threshold_pdf_fn = min(pdfs_fn_source_2)
    elif threshold == 'avg':
        threshold_pdf_tp = np.mean(pdfs_tp_source_1)
        threshold_pdf_fn = np.mean(pdfs_fn_source_2)

    if "shine" in data_type:
        matrix_shine = shine.create_matrix_shine(data1)
        incoming_pdf = np.exp(density_estimator_tp_source_1.score_samples(matrix_shine))[0]
    elif "feature" in data_type:
        incoming_pdf = np.exp(density_estimator_tp_source_1.score_samples(data1))[0]

    if incoming_pdf >= threshold_pdf_tp:
        return False  # no need to intervene, it's probably an TP
    else:
        # inconclusive, let's do a check using a different source
        if "shine" in data_type:
            new_pdf = np.exp(density_estimator_fn_source_2.score_samples(data2))[0]
        elif "feature" in data_type:
            matrix_shine = shine.create_matrix_shine(data2)
            new_pdf = np.exp(density_estimator_fn_source_2.score_samples(matrix_shine))[0]

        if new_pdf >= threshold_pdf_fn:
            return True
        else:
            return False


def predict_with_tp_hybrid(data1, data2, pdfs_tp_source_1, density_estimator_tp_source_1,
                           pdfs_tp_source_2, density_estimator_tp_source_2, data_type, threshold):
    incoming_pdf = None
    new_pdf = None
    threshold_pdf_tp = None
    threshold_pdf_tp_2 = None

    if threshold == 'min':
        threshold_pdf_tp = min(pdfs_tp_source_1)
        threshold_pdf_tp_2 = min(pdfs_tp_source_2)
    elif threshold == 'avg':
        threshold_pdf_tp = np.mean(pdfs_tp_source_1)
        threshold_pdf_tp_2 = np.mean(pdfs_tp_source_2)

    if "shine" in data_type:
        matrix_shine = shine.create_matrix_shine(data1)
        incoming_pdf = np.exp(density_estimator_tp_source_1.score_samples(matrix_shine))[0]
    elif "feature" in data_type:
        incoming_pdf = np.exp(density_estimator_tp_source_1.score_samples(data1))[0]

    if incoming_pdf >= threshold_pdf_tp:
        return False  # no need to intervene, it's probably an TP
    else:
        # inconclusive, let's do a check using a different source
        if "shine" in data_type:
            new_pdf = np.exp(density_estimator_tp_source_2.score_samples(data2))[0]
        elif "feature" in data_type:
            matrix_shine = shine.create_matrix_shine(data2)
            new_pdf = np.exp(density_estimator_tp_source_2.score_samples(matrix_shine))[0]

        if new_pdf <= threshold_pdf_tp_2:
            return True
        else:
            return False


def predict_new_hybrid(data1, data2, pdfs_tp_source_1, density_estimator_tp_source_1,
                       pdfs_fn_source_1, density_estimator_fn_source_1,
                       pdfs_tp_source_2, density_estimator_tp_source_2,
                       pdfs_fn_source_2, density_estimator_fn_source_2, id_ood, pred, label):
    # print("Class:{}----Pred=Label?:{}-----------------------".format(label, pred==label))
    incoming_feature_pdf_tp = np.exp(density_estimator_tp_source_1.score_samples(data1.reshape(1, -1)))[0]
    # print('({}) -- incoming_feature_pdf_tp: {}'.format(id_ood, incoming_feature_pdf_tp))

    # incoming_feature_pdf_fn = np.exp(density_estimator_fn_source_1.score_samples(data1.reshape(1, -1)))[0]

    # print('({}) -- incoming_feature_pdf_fn: {}'.format(id_ood, incoming_feature_pdf_fn))
    # print('({}) -- distance incoming FEATURE TP and FN: {}'.format(
    #    id_ood, incoming_feature_pdf_tp - incoming_feature_pdf_fn))

    # matrix_shine = shine.create_matrix_shine(data2)
    # incoming_shine_pdf_tp = np.exp(density_estimator_tp_source_2.score_samples(matrix_shine))[0]

    # print('({}) -- incoming_shine_pdf_tp: {}'.format(id_ood, incoming_shine_pdf_tp))

    # incoming_shine_pdf_fn = np.exp(density_estimator_fn_source_2.score_samples(matrix_shine))[0]

    # print('({}) -- incoming_shine_pdf_fn: {}'.format(id_ood, incoming_shine_pdf_fn))
    # print('({}) -- distance incoming SHINE TP and FN: {}'.format(
    #    id_ood, incoming_shine_pdf_tp - incoming_shine_pdf_fn))
    # print("\n")

    # if incoming_feature_pdf_tp >= threshold_pdf_tp_1: False
    # elif incoming_shine_pdf_tp <= threshold_pdf_tp_2 / incoming_shine_pdf_fn >= threshold_pdf_fn: True / True
    # else: False
    if incoming_feature_pdf_tp >= np.mean(pdfs_tp_source_1):
        return False  # no need to intervene, it's probably an TP
    else:
        # it is inconclusive if the prediction is TP or FN
        # HINE
        if pdfs_fn_source_2 is not None:
            monitor_pred, pdf = calculate_img_stats(data2, 0.9, pdfs_fn_source_2, density_estimator_fn_source_2)
            return monitor_pred
        else:
            return False


def solve_gaussians(m1, s1, m2, s2):
    x1 = (s1 * s2 * np.sqrt((-2 * np.log(s1 / s2) * s2 ** 2) + 2 * s1 ** 2 * np.log(
        s1 / s2) + m2 ** 2 - 2 * m1 * m2 + m1 ** 2) + m1 * s2 ** 2 - m2 * s1 ** 2) / (s2 ** 2 - s1 ** 2)
    x2 = -(s1 * s2 * np.sqrt((-2 * np.log(s1 / s2) * s2 ** 2) + 2 * s1 ** 2 * np.log(
        s1 / s2) + m2 ** 2 - 2 * m1 * m2 + m1 ** 2) - m1 * s2 ** 2 + m2 * s1 ** 2) / (s2 ** 2 - s1 ** 2)
    return x1, x2


def get_intersec_dists(data1, data2):
    if data2 is not None:
        std_tp = np.std(data1)
        std_fn = np.std(data2)
        mean_tp = np.mean(data1)
        mean_fn = np.mean(data2)
        intersection_dists = solve_gaussians(mean_tp, std_tp, mean_fn, std_fn)
        return intersection_dists
    else:
        return None


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


def calculate_img_stats(X, threshold_SMimg, scores_FN, density_img_fn):
    len_scores = len(scores_FN)
    sorted_scores = sorted(scores_FN)
    id_threshold = int(len_scores * threshold_SMimg)
    threshold = sorted_scores[-id_threshold]

    matrix_shine = shine.create_matrix_shine(X)
    logprob = density_img_fn.score_samples(matrix_shine)

    return np.exp(logprob)[0] <= threshold, np.exp(logprob)


def predict_original(incoming_feature, features_tp_by_class, similarities_TP_by_class, magnitude=1):
    # caculate similarities
    incoming_tp_scores = calculate_feature_similarity(incoming_feature, features_tp_by_class, "single", "euclidean")

    threshold_TP = get_thresholded_data(similarities_TP_by_class, 'new', magnitude)
    threshold_FN = max(similarities_TP_by_class)
    if np.mean(incoming_tp_scores) >= threshold_FN:  # it is probably an FN (incorrect pred)
        return True
    else:
        if np.mean(incoming_tp_scores) <= threshold_TP:
            return False
        else:
            return True


def predict_with_tp_fn(image, pdfs_tp, density_estimator_tp, density_estimator_tp_fn, pdfs_tp_fn):
    min_pdf_tp = min(pdfs_tp)
    matrix_shine = shine.create_matrix_shine(image)
    incoming_pdf = np.exp(density_estimator_tp.score_samples(matrix_shine))
    if incoming_pdf >= min_pdf_tp:
        return False  # no need to intervene, it's probably an TP
    else:
        # it's probably an FN, let's do a last check
        if density_estimator_tp_fn is not None:
            new_pdf = np.exp(density_estimator_tp_fn.score_samples(matrix_shine))
            if new_pdf >= np.average(pdfs_tp_fn):
                return True  # image pdf is similar to pdfs of hard-to-classify FN
        else:
            return True  # There weren't FN pdfs >= TP ones during training, then incoming_pdf is probably an FN


def load_results_test(arr_approachs, id_dataset, ood_type, variant):
    root_path = os.path.join('test', 'results')
    filename_gt = os.path.join(root_path, id_dataset, ood_type, variant, 'ground_truth.sav')
    ground_truth = pickle.load(open(filename_gt, 'rb'))
    results = {}
    for approach in arr_approachs:
        filename = os.path.join(root_path, id_dataset, ood_type, variant, '{}.sav'.format(approach))
        results.update({approach: pickle.load(open(filename, 'rb'))})

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


def load_resources_for_test(root_path, num_cls, filename_format_c, filename_format_i, model):
    # loading density estimators from training
    arr_kde_on_features_tp = {}
    arr_kde_on_features_fn = {}
    arr_kde_on_shine_vectors_tp = {}
    arr_kde_on_shine_vectors_fn = {}

    for cls in range(num_cls):
        path_kde_on_feature_tp = os.path.join(root_path, "on_features", filename_format_c.format(cls))
        kde_on_feature_tp = pickle.load(open(path_kde_on_feature_tp, 'rb'))
        arr_kde_on_features_tp.update({cls: kde_on_feature_tp})

        path_kde_on_feature_fn = os.path.join(root_path, "on_features", filename_format_i.format(cls))
        kde_on_feature_fn = pickle.load(open(path_kde_on_feature_fn, 'rb'))
        arr_kde_on_features_fn.update({cls: kde_on_feature_fn})

        path_kde_on_shine_vectors_tp = os.path.join(root_path, "on_shine_vectors", filename_format_c.format(cls))
        kde_on_shine_vectors_tp = pickle.load(open(path_kde_on_shine_vectors_tp, 'rb'))
        arr_kde_on_shine_vectors_tp.update({cls: kde_on_shine_vectors_tp})

        path_kde_on_shine_vectors_fn = os.path.join(root_path, "on_shine_vectors", filename_format_i.format(cls))
        kde_on_shine_vectors_fn = pickle.load(open(path_kde_on_shine_vectors_fn, 'rb'))
        arr_kde_on_shine_vectors_fn.update({cls: kde_on_shine_vectors_fn})

    # loading pdf scores from training
    arr_pdf_on_features_tp = {}
    arr_pdf_on_features_fn = {}
    arr_pdf_on_shine_vectors_tp = {}
    arr_pdf_on_shine_vectors_fn = {}

    root_path = os.path.join("data_analysis", model, id_dataset)
    filename_format_c = "cls_{}_correct_correct.sav"
    filename_format_i = "cls_{}_correct_incorrect.sav"

    for cls in range(num_cls):
        filename_shine_tp = os.path.join(root_path, "shine_pdf", filename_format_c.format(cls))
        filename_shine_fn = os.path.join(root_path, "shine_pdf", filename_format_i.format(cls))
        filename_feature_tp = os.path.join(root_path, "feature_pdf", filename_format_c.format(cls))
        filename_feature_fn = os.path.join(root_path, "feature_pdf", filename_format_i.format(cls))

        scores_shine_pdf_tp = pickle.load(open(filename_shine_tp, 'rb'))
        scores_shine_pdf_fn = pickle.load(open(filename_shine_fn, 'rb'))
        scores_feature_pdf_tp = pickle.load(open(filename_feature_tp, 'rb'))
        scores_feature_pdf_fn = pickle.load(open(filename_feature_fn, 'rb'))

        arr_pdf_on_features_tp.update({cls: scores_feature_pdf_tp})
        arr_pdf_on_features_fn.update({cls: scores_feature_pdf_fn})
        arr_pdf_on_shine_vectors_tp.update({cls: scores_shine_pdf_tp})
        arr_pdf_on_shine_vectors_fn.update({cls: scores_shine_pdf_fn})

    return arr_kde_on_features_tp, arr_kde_on_features_fn, arr_kde_on_shine_vectors_tp, arr_kde_on_shine_vectors_fn, \
           arr_pdf_on_features_tp, arr_pdf_on_features_fn, arr_pdf_on_shine_vectors_tp, arr_pdf_on_shine_vectors_fn


def main(model, layer_relu_ids, percentage_repr_tp, id_dataset, magnitude, num_classes, ood_dataset, transform):
    # model params
    batch_size = 100

    # monitor params
    id_layer_monitored = -1

    # file params
    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
        dataset_train)
    x_test, features_test, logits_test, softmax_test, pred_test, lab_test = feature_extractor.get_features(
        dataset_test)

    dataset_ood = Dataset(ood_dataset, "train", model, batch_size=batch_size, additional_transform=transform)

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
    for c in range(num_classes):
        k = int(len(arr_pdf_on_features_tp[
                        c]) / 100 * percentage_repr_tp)  # we want the top 5% representative TPs of each class
        # print("getting the {}th most representative true positives of the class {}".format(k, c))
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
        features_tp_by_class.update({c: features_tp})
        scores_TP = calculate_feature_similarity(features_tp, features_tp, "tp", "euclidean")
        similarities_TP_by_class.update({c: scores_TP})

        if len(ind_incorrect_ML) > 1:
            features_fn = features_train[list(ind_incorrect_ML)]
            scores_FN = calculate_feature_similarity(features_fn, features_tp, "fn", "euclidean")

            features_fn_by_class.update({c: features_fn})
            similarities_FN_by_class.update({c: scores_FN})

        else:
            features_fn_by_class.update({c: None})
            similarities_FN_by_class.update({c: None})

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

        if pred == label:  # monitor does not need to activate
            m_true.append(False)
            m_ood.append(False)
        else:  # monitor should activate
            m_true.append(True)
            m_ood.append(True)

        data_source_1 = feature

        result = predict_original(data_source_1, features_tp_by_class[pred],
                                  similarities_TP_by_class[pred], magnitude)
        monitor_results.append(result)
        monitor_ood_results.append(result)

    print("OOD MCC: {} and confusion matrix: {}".format(mcc(y_true=m_ood, y_pred=monitor_ood_results),
                                                        confusion_matrix(m_ood, monitor_ood_results)))
    return monitor_results, m_true


if __name__ == '__main__':
    # dataset params
    ood_type = "distributional_shift"
    additional_transforms = ["blur", "brightness", "pixelization", "shuffle_pixels",
                             "contrast", "opacity", "rotate", "saturation"]

    id_dataset = "cifar10"  # "svhn" "cifar10"
    magnitude = 2  # 2 for CIFAR; 1 for SVHN
    model = "resnet"  # "resnet"
    layer_relu_ids = [32]  # 32
    ood_dataset = id_dataset
    num_classes = 10
    arr_results = []
    arr_approach_names = ['sena']

    percentage_repr_tp = 10  # the percentage of instances to be kept as representative TP at runtime

    for transform in additional_transforms:
        m_true = None
        for approach in arr_approach_names:
            monitor_results, m_true = main(model, layer_relu_ids, percentage_repr_tp, id_dataset, magnitude,
                                           num_classes, ood_dataset, transform)
            # saving results
            arr_results.append(monitor_results)

        save_test_results(arr_results, arr_approach_names, id_dataset, ood_type, transform, m_true)

        ground_truth, results = load_results_test(arr_approach_names, id_dataset, ood_type, transform)

        utils.print_results_test(id_dataset, ood_type, arr_approach_names, results, ground_truth, transform)
