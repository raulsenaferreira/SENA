# external libs
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix
from sklearn.metrics import matthews_corrcoef as mcc
# internal libs
from dataset import Dataset
from feature_extractor import FeatureExtractor
from methods import shine


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
    threshold_pdf_fn = None

    if threshold == 'min':
        threshold_pdf_tp = min(pdfs_tp_source_1)
        threshold_pdf_fn = min(pdfs_tp_source_2)
    elif threshold == 'avg':
        threshold_pdf_tp = np.mean(pdfs_tp_source_1)
        threshold_pdf_fn = np.mean(pdfs_tp_source_2)

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

        if new_pdf <= threshold_pdf_fn:
            return True
        else:
            return False


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


def print_results_test(arr_approach, arr_results, ground_truth, variant):
    root_path = os.path.join('test', 'results')
    evaluation_text = ""

    for approach in arr_approach:
        tn, fp, fn, tp = confusion_matrix(ground_truth, arr_results[approach]).ravel()
        sg_score = tp / len(ground_truth)
        rh_score = fn / len(ground_truth)
        ac_score = fp / len(ground_truth)
        evaluation_text += "{}" \
                           "\nMCC: {}" \
                           "\nBalanced accuracy: {}" \
                           "\nFPR: {}" \
                           "\nFNR: {}" \
                           "\nPrecision: {}" \
                           "\nRecall: {}" \
                           "\nF1: {}" \
                           "\nSG: {}" \
                           "\nRH: {}" \
                           "\nAC: {}" \
                           "\n\n\n".format(approach.upper(),
                                           mcc(ground_truth, arr_results[approach]),
                                           balanced_accuracy_score(ground_truth, arr_results[approach]),
                                           fp / (fp + tn),
                                           fn / (tp + fn),
                                           precision_score(ground_truth, arr_results[approach]),
                                           recall_score(ground_truth, arr_results[approach]),
                                           f1_score(ground_truth, arr_results[approach]),
                                           sg_score,
                                           rh_score,
                                           ac_score)

        # print(classification_report(ground_truth, results[monitor_name]))

        # saving in a txt file for posterior analysis
        with open(os.path.join(root_path, id_dataset, ood_type, variant, "evaluation_output.txt"), "w") as text_file:
            text_file.write(evaluation_text)


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


def main(approach, id_dataset, num_classes, ood_dataset, transform):
    # model params
    batch_size = 100
    model = "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]

    # file params
    root_path = os.path.join("methods", "fitted_kde", model, id_dataset)
    filename_format_c = "kde_cls_{}_tp.sav"
    filename_format_i = "kde_cls_{}_fn.sav"

    # dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    # x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
    #    dataset_train)
    x_test, features_test, logits_test, softmax_test, pred_test, lab_test = feature_extractor.get_features(
        dataset_test)

    dataset_ood = Dataset(ood_dataset, "test", model, additional_transform=transform, batch_size=batch_size)

    x_ood, features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

    # accuracy of the ML model
    id_accuracy = accuracy_score(lab_test, pred_test)
    ood_accuracy = 0
    if id_dataset == ood_dataset:
        ood_accuracy = accuracy_score(lab_ood, pred_ood)
    print("Model accuracy")
    print("ID:  ", id_accuracy)
    print("OOD: ", ood_accuracy)

    arr_kde_on_features_tp, arr_kde_on_features_fn, arr_kde_on_shine_vectors_tp, arr_kde_on_shine_vectors_fn, \
    arr_pdf_on_features_tp, arr_pdf_on_features_fn, arr_pdf_on_shine_vectors_tp, arr_pdf_on_shine_vectors_fn = \
        load_resources_for_test(root_path, num_classes, filename_format_c, filename_format_i, model)

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

        data_source_1 = None
        data_source_2 = None
        arr_pdf_tp_source_1 = None
        arr_kde_tp_source_1 = None
        arr_pdf_fn_source_2 = None
        arr_kde_fn_source_2 = None

        if "shine" in approach:
            data_source_1 = np.array([x])
            data_source_2 = np.array([feature])
            arr_pdf_tp_source_1 = arr_pdf_on_shine_vectors_tp[pred]
            arr_pdf_fn_source_2 = arr_pdf_on_features_fn[pred]
            arr_kde_tp_source_1 = arr_kde_on_shine_vectors_tp[pred]
            arr_kde_fn_source_2 = arr_kde_on_features_fn[pred]
        elif "feature" in approach:
            data_source_1 = np.array([feature])
            data_source_2 = np.array([x])
            arr_pdf_tp_source_1 = arr_pdf_on_features_tp[pred]
            arr_pdf_fn_source_2 = arr_pdf_on_shine_vectors_fn[pred]
            arr_kde_tp_source_1 = arr_kde_on_features_tp[pred]
            arr_kde_fn_source_2 = arr_kde_on_shine_vectors_fn[pred]

        threshold_type = None
        if "min" in approach:
            threshold_type = "min"
        elif "avg" in approach:
            threshold_type = "avg"

        if "hybrid" in approach:
            # hybrid approach
            result = predict_with_tp_fn_hybrid(data_source_1, data_source_2, arr_pdf_tp_source_1,
                                               arr_kde_tp_source_1, arr_pdf_fn_source_2, arr_kde_fn_source_2,
                                               approach, threshold_type)
            monitor_results.append(result)
        else:
            # approach TP
            result = predict_with_tp(data_source_1, arr_pdf_tp_source_1, arr_kde_tp_source_1,
                                     approach, threshold_type)
            monitor_results.append(result)

    # dist shift data
    for x, feature, softmax, logits, pred, label in tqdm(
            zip(x_ood, features_ood[id_layer_monitored], softmax_ood,
                logits_ood, pred_ood, lab_ood)):

        if pred == label:  # monitor does not need to activate
            m_true.append(False)
        else:  # monitor should activate
            m_true.append(True)

        data_source_1 = None
        data_source_2 = None
        arr_pdf_tp_source_1 = None
        arr_pdf_tp_source_2 = None
        arr_kde_tp_source_1 = None
        arr_kde_tp_source_2 = None
        arr_pdf_fn_source_2 = None
        arr_kde_fn_source_2 = None

        if "shine" in approach:
            data_source_1 = np.array([x])
            data_source_2 = np.array([feature])
            arr_pdf_tp_source_1 = arr_pdf_on_shine_vectors_tp[pred]
            arr_pdf_tp_source_2 = arr_pdf_on_features_tp[pred]
            arr_pdf_fn_source_2 = arr_pdf_on_features_fn[pred]
            arr_kde_tp_source_1 = arr_kde_on_shine_vectors_tp[pred]
            arr_kde_tp_source_2 = arr_kde_on_features_tp[pred]
            arr_kde_fn_source_2 = arr_kde_on_features_fn[pred]
        elif "feature" in approach:
            data_source_1 = np.array([feature])
            data_source_2 = np.array([x])
            arr_pdf_tp_source_1 = arr_pdf_on_features_tp[pred]
            arr_pdf_tp_source_2 = arr_pdf_on_shine_vectors_tp[pred]
            arr_pdf_fn_source_2 = arr_pdf_on_shine_vectors_fn[pred]
            arr_kde_tp_source_1 = arr_kde_on_features_tp[pred]
            arr_kde_tp_source_2 = arr_kde_on_shine_vectors_tp[pred]
            arr_kde_fn_source_2 = arr_kde_on_shine_vectors_fn[pred]

        threshold_type = None
        if "min" in approach:
            threshold_type = "min"
        elif "avg" in approach:
            threshold_type = "avg"

        if "hybrid" in approach:
            # hybrid approach
            result = predict_with_tp_hybrid(data_source_1, data_source_2, arr_pdf_tp_source_1,
                                            arr_kde_tp_source_1, arr_pdf_tp_source_2, arr_kde_tp_source_2,
                                            approach, threshold_type)
            # predict_with_tp_fn_hybrid(data_source_1, data_source_2, arr_pdf_tp_source_1,
            #                                   arr_kde_tp_source_1, arr_pdf_fn_source_2, arr_kde_fn_source_2,
            #                                   approach, threshold_type)
            monitor_results.append(result)
        else:
            # approach TP
            result = predict_with_tp(data_source_1, arr_pdf_tp_source_1, arr_kde_tp_source_1,
                                     approach, threshold_type)
            monitor_results.append(result)

    return monitor_results, m_true


if __name__ == '__main__':
    # dataset params
    ood_type = "distributional_shift"
    additional_transforms = ["blur", "brightness", "pixelization", "shuffle_pixels"]
    id_dataset = "cifar10"  # "svhn" #
    ood_dataset = "cifar10"  # "cifar10"#
    num_classes = 10
    arr_results = []
    arr_approach_names = [  # 'shine_min', 'feature_min', 'shine_avg', 'feature_avg',
        # 'shine_hybrid_min', 'shine_hybrid_avg', 'feature_hybrid_min',
        'feature_min', 'feature_avg', 'feature_hybrid_avg']
    for transform in additional_transforms:
        m_true = None
        for approach in arr_approach_names:
            monitor_results, m_true = main(approach, id_dataset, num_classes, ood_dataset, transform)
            # saving results
            arr_results.append(monitor_results)

        save_test_results(arr_results, arr_approach_names, id_dataset, ood_type, transform, m_true)

        ground_truth, results = load_results_test(arr_approach_names, id_dataset, ood_type, transform)

        print_results_test(arr_approach_names, results, ground_truth, transform)
