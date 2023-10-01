import os
from random import randint
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef as mcc
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
from keras import backend as K
from keras.datasets import mnist, fashion_mnist, cifar10
from methods import shine
import seaborn as sns


def plot_dist_fn_tp(data1, data2):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    b = sns.distplot(data1, kde=True, ax=ax, hist=False, bins=10, kde_kws=dict(linewidth=5))
    b = sns.distplot(data2, kde=True, ax=ax, hist=False, bins=10, kde_kws=dict(linewidth=5))
    # plt.xlabel('Euclidean distance')
    b.set_xlabel("Euclidean distance", fontsize=18)
    b.set_ylabel("Density", fontsize=18)
    b.tick_params(labelsize=18)
    plt.show()


def load_svhn(print_img=False):
    train_path = os.path.join('Data', 'train_32x32.mat')
    test_path = os.path.join('Data', 'test_32x32.mat')

    train_raw, test_raw = None, None

    try:
        train_raw = loadmat(train_path)
    except:
        print('file {} not found'.format(train_path))
    try:
        test_raw = loadmat(test_path)
    except:
        print('file {} not found'.format(test_path))

    train_images, train_labels = None, None
    # Load images and labels
    if train_raw is not None:
        train_images = np.array(train_raw['X'])
        train_labels = train_raw['y']
        # Fix the axes of the images
        train_images = np.moveaxis(train_images, -1, 0)
        print(train_images.shape)

    test_images, test_labels = None, None
    if test_raw is not None:
        test_images = np.array(test_raw['X'])
        test_labels = test_raw['y']
        # Fix the axes of the images
        test_images = np.moveaxis(test_images, -1, 0)
        print(test_images.shape)

    # Plot a random image and its label
    if print_img:
        random_index = randint(0, 10000)
        plt.imshow(train_images[random_index])
        plt.show()
        print('Label: ', train_labels[random_index])

    # # Convert train and test images into 'float64' type
    # train_images = train_images.astype('float64')
    # test_images = test_images.astype('float64')

    # # Convert train and test labels into 'int64' type
    # train_labels = train_labels.astype('int64')
    # test_labels = test_labels.astype('int64')

    # # Normalize the images data
    # print('Min: {}, Max: {}'.format(train_images.min(), train_images.max()))

    # train_images /= 255.0
    # test_images /= 255.0

    return (train_images, train_labels), (test_images, test_labels)


def load_data(dataset_name, num_samples=None):
    # the data, split between train and test sets
    if dataset_name == 'mnist':
        img_rows, img_cols, dim = 28, 28, 1
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        img_rows, img_cols, dim = 28, 28, 1
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    elif dataset_name == 'cifar10':
        img_rows, img_cols, dim = 32, 32, 3
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset_name == 'svhn':
        img_rows, img_cols, dim = 32, 32, 3
        (x_train, y_train), (x_test, y_test) = load_svhn()
    else:
        print('No dataset available...')
        return None

    # Limit the data to n samples
    if num_samples is not None:
        x_train = x_train[:num_samples]
        y_train = y_train[:num_samples]
        x_test = x_test[:num_samples]
        y_test = y_test[:num_samples]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], dim, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], dim, img_rows, img_cols)
        input_shape = (dim, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, dim)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, dim)
        input_shape = (img_rows, img_cols, dim)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    return (x_train, y_train), (x_test, y_test)


def convert_ood_labels(y_test, cls=None):
    # all labels are OOD by default
    y_true = [1] * len(y_test)

    y_true = np.array(y_true)

    if cls != None:
        ind = np.where(y_test == cls)[0]
        y_true[ind] = 0

    # print('y_true',y_true)
    return y_true


# Function returns the mean of values between quartiles of scores calculated for correct classified data
def calculate_threshold(monitor, model_internals, y=None, score_perc=0.95):
    score = None

    if monitor.name == 'react' or monitor.name == 'msp' or monitor.name == 'maxlogit' or monitor.name == 'energy':
        score = monitor.predict(model_internals)  # (np.array([internal_val]))[0]

    elif monitor.name == 'mahalanobis':
        score = monitor.predict(model_internals,
                                np.array([y] * len(model_internals)))  # (np.array([internal_val]), [pred])[0]

        # print(monitor.name, score)

        # arr_scores.append(score)

    # get the threshold
    arr_scores = np.array(score)  # (arr_scores)
    arr_scores.sort()
    limit = int(len(arr_scores) * score_perc)

    return arr_scores[limit]

'''
# Finding the best thresholds for some family of monitors using just the training set as reference
def calculate_optimal_threshold(monitor, model_internals, labels):
    score = None

    if monitor.name == 'react' or monitor.name == 'msp' or monitor.name == 'maxlogit' or monitor.name == 'energy':
        score = monitor.predict(model_internals)

    elif monitor.name == 'mahalanobis':
        score = monitor.predict(model_internals, np.array([labels] * len(model_internals)))

        # print(monitor.name, score)

        # arr_scores.append(score)

    for
    # get the threshold
    arr_scores = np.array(score)  # (arr_scores)
    arr_scores.sort()
    limit = int(len(arr_scores) * score_perc)

    return arr_scores[limit]
'''


def calculate_threshold_SHINE(monitor, misclass_images, label, predictions, model_internals, score_perc=0.95):
    threshold_S = {}

    # indices correct pred
    ind_ML_cor = np.where(predictions == label)[0]
    # print('len pred class',label, np.shape(ind_ML_cor), ind_ML_cor)

    # features from correct predictions
    f_c_correct = model_internals[ind_ML_cor]

    # indices INcorrect pred
    ind_ML_inc = np.where(predictions != label)[0]
    # print('len pred class',label, np.shape(ind_ML_inc), ind_ML_inc)

    # features from INcorrect predictions
    f_c_incorrect = model_internals[ind_ML_inc]

    # calculating threshold for TP and FN
    similarities_tp = []
    similarities_fn = []

    for rf in monitor.arr_repr_feature_TP[label]:

        for df_c in f_c_correct:
            cosine_similarity = 1 - spatial.distance.cosine(rf, df_c)
            similarities_tp.append(cosine_similarity)

        for df_i in f_c_incorrect:
            cosine_similarity = 1 - spatial.distance.cosine(rf, df_i)
            similarities_fn.append(cosine_similarity)

    threshold_S['TP'] = np.sum(similarities_tp) / len(similarities_tp)
    threshold_S['FN'] = np.sum(similarities_fn) / len(similarities_fn)

    # calculating threshold for HINE
    matrix_shine = shine.create_matrix_shine(misclass_images)
    arr_scores_hine = np.exp(monitor.arr_density_img[label].score_samples(matrix_shine))

    # get the threshold
    arr_scores_hine.sort()
    limit_hine = int(len(arr_scores_hine) * score_perc)

    return threshold_S, arr_scores_hine[limit_hine]


import gzip


def load_dataset(threat_type, variation_type, dataset_name, mode, val_perc=0.2, root_path='data'):
    x_train, y_train, x_test, y_test = None, None, None, None

    if mode == 'test':
        fixed_path = os.path.join(root_path, 'benchmark_dataset', threat_type, variation_type)
        if dataset_name != None:
            fixed_path = os.path.join(root_path, 'benchmark_dataset', threat_type, dataset_name, variation_type)
        print('loading data from', fixed_path)
        test_images = os.path.join(fixed_path, 'test-images-npy.gz')
        test_labels = os.path.join(fixed_path, 'test-labels-npy.gz')

        f = gzip.GzipFile(test_images, "r")
        x_test = np.load(f)

        f = gzip.GzipFile(test_labels, "r")
        y_test = np.load(f)

        return x_test, y_test

    else:
        fixed_path = os.path.join(root_path, 'training_set', threat_type, variation_type)
        if dataset_name != None:
            fixed_path = os.path.join(root_path, 'training_set', threat_type, dataset_name, variation_type)

        print('loading data from', fixed_path)
        train_images = os.path.join(fixed_path, 'train-images-npy.gz')
        train_labels = os.path.join(fixed_path, 'train-labels-npy.gz')

        f = gzip.GzipFile(train_images, "r")
        x_train = np.load(f)

        f = gzip.GzipFile(train_labels, "r")
        y_train = np.load(f)

        if mode == 'train_val':
            size_val = int(len(y_train) * val_perc)
            return (x_train[:size_val], y_train[:size_val]), (x_train[size_val:], y_train[size_val:])

        else:
            return x_train, y_train


from scipy import spatial
import pickle

filename_c = 'Features/RESNET/cifar10_{}_correct.sav'
filename_i = 'Features/RESNET/cifar10_{}_incorrect.sav'


def similarity_act_func(features, id_y_train, pred_train):
    for cls in range(10):
        scores_1 = []
        scores_2 = []

        # all labels from class c
        ind_y_c = np.where(id_y_train == cls)[0]
        # print('len all labels class',c, np.shape(ind_y_c), ind_y_c)

        # all pred as c
        ind_ML_c = np.where(pred_train == cls)[0]
        # print('len pred class',c, np.shape(ind_ML_c), ind_ML_c)

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
                scores_1.append(cosine_similarity)

        for i in range(1, len(f_c_correct)):
            cosine_similarity = 1 - spatial.distance.cosine(f_c_correct[i - 1], f_c_correct[i])
            scores_2.append(cosine_similarity)

        pickle.dump(scores_1, open(filename_i.format(cls), 'wb'))
        pickle.dump(scores_2, open(filename_c.format(cls), 'wb'))

        print('sim correct/incorrect pred', np.sum(scores_1) / len(scores_1))
        print('sim between pairs de correct pred', np.sum(scores_2) / len(scores_2))


# saving monitors
def save_monitors(arr_monitors, id_dataset):
    for monitor in arr_monitors:
        filename = os.path.join('Monitors', id_dataset, '{}.sav'.format(monitor.name))
        print("saving monitor {}".format(monitor.name))
        pickle.dump(monitor, open(filename, 'wb'))


# loading monitors
def load_monitor(id_dataset, monitor_name):
    monitor_filename = os.path.join('Monitors', id_dataset, '{}.sav'.format(monitor_name))
    monitor = pickle.load(open(monitor_filename, 'rb'))
    return monitor


def save_results(arr_monitors, id_dataset, ood_type, variant, ground_truth):
    filename_gt = os.path.join('results', id_dataset, ood_type, variant, 'ground_truth.sav')
    pickle.dump(ground_truth, open(filename_gt, 'wb'))

    for monitor in arr_monitors:
        if monitor is not None:
            filename = os.path.join('results', id_dataset, ood_type, variant, '{}.sav'.format(monitor.name))
            print("saving results from {} for {}".format(monitor.name, ood_type))
            pickle.dump(monitor.results, open(filename, 'wb'))


def load_results(arr_monitor_names, id_dataset, ood_type, variant):
    filename_gt = os.path.join('results', id_dataset, ood_type, variant, 'ground_truth.sav')
    ground_truth = pickle.load(open(filename_gt, 'rb'))
    results = {}
    for monitor_name in arr_monitor_names:
        filename = os.path.join('results', id_dataset, ood_type, variant, '{}.sav'.format(monitor_name))
        print("Loading results from {} for {}".format(monitor_name, ood_type))
        results.update({monitor_name: pickle.load(open(filename, 'rb'))})

    return ground_truth, results


def load_sena_results(id_dataset, ood_type, variant):
    filename_gt = os.path.join('results', id_dataset, ood_type, variant, 'ground_truth.sav')
    ground_truth = pickle.load(open(filename_gt, 'rb'))
    name = 'sena'
    filename = os.path.join('test', 'results', id_dataset, ood_type, variant, '{}.sav'.format(name))
    print("Loading results from {} for {}".format(name, ood_type))
    results = pickle.load(open(filename, 'rb'))

    return ground_truth, results


def load_baseline_results(id_dataset, ood_type, variant):
    filename = os.path.join('results', id_dataset, ood_type, variant, 'baseline.sav')
    print("Loading results from Baseline for {}".format(ood_type))
    ground_truth, results = pickle.load(open(filename, 'rb'))

    return ground_truth, results


def save_baseline_results(ood_type, ood_variant, id_dataset, ood_dataset, pred_test, lab_test, pred_ood, lab_ood):
    id_accuracy = mcc(lab_test, pred_test)
    ood_accuracy = 0
    if id_dataset == ood_dataset:
        ood_accuracy = mcc(lab_ood, pred_ood)
    print("Model MCC")
    print("ID: ", id_accuracy)
    print("OOD: ", ood_accuracy)

    ml_res = np.hstack([pred_test, pred_ood])
    gr_ood = lab_ood
    if id_dataset != ood_dataset:
        gr_ood = np.array([-1] * len(lab_ood))

    gr_tr = np.hstack([lab_test, gr_ood])
    print("ID+OOD: ", mcc(gr_tr, ml_res))

    # saving results for posterior analysis
    filename = os.path.join('results', id_dataset, ood_type, ood_variant, 'baseline.sav')
    print("saving results from baseline for {}".format(ood_type))
    pickle.dump((gr_tr, ml_res), open(filename, 'wb'))

    evaluation_text = ""
    CF = confusion_matrix(gr_tr, ml_res)
    fp = CF.sum(axis=0) - np.diag(CF)
    fn = CF.sum(axis=1) - np.diag(CF)
    tp = np.diag(CF)
    tn = CF.sum() - (fp + fn + tp)
    sg_score = tp / len(gr_tr)
    rh_score = fn / len(gr_tr)
    ac_score = fp / len(gr_tr)
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
                       "\n\n\n".format("BASELINE",
                                       mcc(gr_tr, ml_res),
                                       balanced_accuracy_score(gr_tr, ml_res),
                                       fp / (fp + tn),
                                       fn / (tp + fn),
                                       precision_score(gr_tr, ml_res, average='macro'),
                                       recall_score(gr_tr, ml_res, average='macro'),
                                       f1_score(gr_tr, ml_res, average='macro'),
                                       sg_score,
                                       rh_score,
                                       ac_score)

    # saving in a txt file for posterior analysis
    with open(os.path.join("csv", id_dataset, ood_type, ood_variant, "baseline_evaluation.txt"),
              "w") as text_file:
        text_file.write(evaluation_text)


def print_results_test_single(id_dataset, ood_type, approach, arr_results, ground_truth, variant):
    root_path = os.path.join('test', 'results')
    evaluation_text = ""

    tn, fp, fn, tp = confusion_matrix(ground_truth, arr_results).ravel()
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
                       "\nMacro F1: {}" \
                       "\nSG: {}" \
                       "\nRH: {}" \
                       "\nAC: {}" \
                       "\n\n\n".format(approach.upper(),
                                       mcc(ground_truth, arr_results),
                                       balanced_accuracy_score(ground_truth, arr_results),
                                       fp / (fp + tn),
                                       fn / (tp + fn),
                                       precision_score(ground_truth, arr_results, average='macro'),
                                       recall_score(ground_truth, arr_results, average='macro'),
                                       f1_score(ground_truth, arr_results, average='macro'),
                                       sg_score,
                                       rh_score,
                                       ac_score)

    # print(classification_report(ground_truth, results[monitor_name]))

    # saving in a txt file for posterior analysis
    with open(os.path.join(root_path, id_dataset, ood_type, variant, "evaluation_output.txt"), "w") as text_file:
        text_file.write(evaluation_text)


def print_results_test(id_dataset, ood_type, arr_approach, arr_results, ground_truth, variant):
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
                           "\nMacro F1: {}" \
                           "\nSG: {}" \
                           "\nRH: {}" \
                           "\nAC: {}" \
                           "\n\n\n".format(approach.upper(),
                                           mcc(ground_truth, arr_results[approach]),
                                           balanced_accuracy_score(ground_truth, arr_results[approach]),
                                           fp / (fp + tn),
                                           fn / (tp + fn),
                                           precision_score(ground_truth, arr_results[approach], average='macro'),
                                           recall_score(ground_truth, arr_results[approach], average='macro'),
                                           f1_score(ground_truth, arr_results[approach], average='macro'),
                                           sg_score,
                                           rh_score,
                                           ac_score)

        # print(classification_report(ground_truth, results[monitor_name]))

        # saving in a txt file for posterior analysis
        with open(os.path.join(root_path, id_dataset, ood_type, variant, "evaluation_output.txt"), "w") as text_file:
            text_file.write(evaluation_text)