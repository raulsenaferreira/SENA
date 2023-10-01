# external libs
import os
from scipy import spatial
import pickle
import numpy as np
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
# internal libs
from dataset import Dataset
from feature_extractor import FeatureExtractor
from methods import shine


def fit_density_estimator(data):
    density_estimator_feature = shine.calculate_density(data, CV_num=3)

    return density_estimator_feature


def calculate_similarity(arr1, arr2, type="pairwise"):
    scores = []
    if type == "pairwise":
        for i in range(len(arr1) - 1):
            cosine_similarity = 1 - spatial.distance.cosine(arr1[i], arr1[i + 1])
            scores.append(cosine_similarity)
    else:
        for v1 in arr1:
            for v2 in arr2:
                cosine_similarity = 1 - spatial.distance.cosine(v2, v1)
                scores.append(cosine_similarity)
    return scores


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


def generate_and_save_scores(num_classes, data_type, x_train, features, id_y_train, pred_train, root_path,
                             filename_format_c, filename_format_i):
    for cls in range(num_classes):
        print("Generating and saving {} data for class {}:".format(data_type, cls))
        pred_c_correct = None
        pred_c_incorrect = None

        scores_correct = None
        scores_incorrect = None

        filename_c = os.path.join(root_path, data_type, filename_format_c.format(cls))
        filename_i = os.path.join(root_path, data_type, filename_format_i.format(cls))

        ind_tp, ind_fn = get_indices_correct_incorrect_predictions(cls, id_y_train, pred_train)

        if "feature" in data_type:
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

        if "similarity" in data_type:
            # cosine distance between act functions triggered during correct predictions only
            scores_correct = calculate_similarity(pred_c_correct, pred_c_correct)
            # cosine distance between act functions triggered during incorrect and correct predictions
            scores_incorrect = calculate_similarity(pred_c_incorrect, pred_c_correct, type="dot_product")
        elif "pdf" in data_type:
            # PDFs generated from correct predictions on a density estimator fitted with correct predictions
            print("pred_c_correct from class {}".format(cls), len(pred_c_correct))
            kde_feature_c = fit_density_estimator(pred_c_correct)
            scores_correct = np.exp(kde_feature_c.score_samples(pred_c_correct))
            # PDfs generated from incorrect predictions on a density estimator fitted with correct predictions
            print("pred_c_incorrect from class {}".format(cls), len(pred_c_incorrect))
            if len(pred_c_incorrect) > 1:
                scores_incorrect = np.exp(kde_feature_c.score_samples(pred_c_incorrect))

        if scores_correct is not None:
            pickle.dump(scores_correct, open(filename_c, 'wb'))
        if scores_incorrect is not None:
            pickle.dump(scores_incorrect, open(filename_i, 'wb'))


def plot_analysis(num_classes, data_type, root_path, filename_format_c, filename_format_i):
    for cls in range(num_classes):
        filename_c = os.path.join(root_path, data_type, filename_format_c.format(cls))
        filename_i = os.path.join(root_path, data_type, filename_format_i.format(cls))
        scores_correct = pickle.load(open(filename_c, 'rb'))
        scores_incorrect = pickle.load(open(filename_i, 'rb'))

        # getting thresholds based on quartiles for FN
        # correct pred
        bp_incoming_X = plt.boxplot(scores_correct, showmeans=True)
        mean_cor = list(set(bp_incoming_X['means'][0].get_ydata()))
        boxes_cor = list(set(bp_incoming_X['boxes'][0].get_ydata()))
        min_Q1_Q3_cor = min(boxes_cor)
        max_Q1_Q3_cor = max(boxes_cor)
        mean_Q1_Q3_cor = (min_Q1_Q3_cor + max_Q1_Q3_cor) / 2

        # incorrect pred
        bp_incoming_X = plt.boxplot(scores_incorrect, showmeans=True)
        mean_incor = list(set(bp_incoming_X['means'][0].get_ydata()))
        boxes_incor = list(set(bp_incoming_X['boxes'][0].get_ydata()))
        min_Q1_Q3_inc = min(boxes_incor)
        max_Q1_Q3_inc = max(boxes_incor)
        mean_Q1_Q3_inc = (min_Q1_Q3_inc + max_Q1_Q3_inc) / 2

        print("-----------------------------------------------------------------")
        print("--> {} analysis (Correct predictions) for class {}".format(data_type, cls))
        # print('avg sim between pairs de correct for class {}'.format(cls), np.sum(scores_correct)/len(scores_correct))
        print('min_Q1/max_Q3 correct:', min_Q1_Q3_cor, max_Q1_Q3_cor)
        print('avg correct:', mean_cor)  # same of average
        print('avg of quartiles Q1 and Q3 correct:', mean_Q1_Q3_cor)
        print("\n")
        print("--> {} analysis (Incorrect predictions) for class {}".format(data_type, cls))
        # print('avg sim correct/incorrect for class {}'.format(cls), np.sum(scores_incorrect) / len(scores_incorrect))
        print('min_Q1/max_Q3 incorrect/correct:', mean_Q1_Q3_inc, max_Q1_Q3_inc)
        print('avg incorrect/correct:', mean_incor)  # same of average
        print('avg of quartiles Q1 and Q3 incorrect/correct:', mean_Q1_Q3_inc)
        print("-----------------------------------------------------------------")
        print("\n\n")


def plot_intersection_analysis_feature2shine(num_classes, imgs, root_path, filename_format_c, filename_format_i):
    for cls in range(num_classes):
        print("Analysis for class: {}".format(cls))

        filename_feature_c = os.path.join(root_path, "feature_pdf", filename_format_c.format(cls))
        filename_feature_i = os.path.join(root_path, "feature_pdf", filename_format_i.format(cls))
        filename_shine_c = os.path.join(root_path, "shine_pdf", filename_format_c.format(cls))

        scores_feature_pdf_correct = pickle.load(open(filename_feature_c, 'rb'))
        scores_feature_pdf_incorrect = pickle.load(open(filename_feature_i, 'rb'))
        scores_shine_pdf_correct = pickle.load(open(filename_shine_c, 'rb'))

        print("len(scores_feature_pdf_correct)", len(scores_feature_pdf_correct))
        print("len(scores_feature_pdf_incorrect)", len(scores_feature_pdf_incorrect))
        print("len(scores_shine_pdf_correct)", len(scores_shine_pdf_correct))

        min_score_feature_pdf_correct = min(scores_feature_pdf_correct)
        indices_score_feature_pdf_intersection = []
        indices_score_shine_pdf_intersection = []

        for i in range(len(scores_feature_pdf_incorrect)):
            if scores_feature_pdf_incorrect[i] >= min_score_feature_pdf_correct:
                indices_score_feature_pdf_intersection.append(i)

        if len(indices_score_feature_pdf_intersection) > 0:
            selected_imgs = imgs[indices_score_feature_pdf_intersection]
            matrix_shine = shine.create_matrix_shine(selected_imgs)
            density_estimator = fit_density_estimator(matrix_shine)
            scores_shine_pdf_incorrect_intersection = density_estimator.score_samples(matrix_shine)

            # analyzing if we can find better separation between FN and TP now
            min_pdf_shine_correct = min(scores_shine_pdf_correct)

            for i in range(len(scores_shine_pdf_incorrect_intersection)):
                if scores_shine_pdf_incorrect_intersection[i] >= min_pdf_shine_correct:
                    indices_score_shine_pdf_intersection.append(i)

        # let's see if the number of difficult scores to classify as Fn or Tp was improved
        print("before:{}; after:{}\n".format(
            len(indices_score_feature_pdf_intersection),
            len(indices_score_shine_pdf_intersection))
        )


def plot_intersection_analysis_shine2feature(num_classes, features, root_path, filename_format_c, filename_format_i):
    for cls in range(num_classes):
        print("Analysis for class: {}".format(cls))

        filename_shine_c = os.path.join(root_path, "shine_pdf", filename_format_c.format(cls))
        filename_shine_i = os.path.join(root_path, "shine_pdf", filename_format_i.format(cls))
        filename_feature_c = os.path.join(root_path, "feature_pdf", filename_format_c.format(cls))

        scores_shine_pdf_correct = pickle.load(open(filename_shine_c, 'rb'))
        scores_shine_pdf_incorrect = pickle.load(open(filename_shine_i, 'rb'))
        scores_feature_pdf_correct = pickle.load(open(filename_feature_c, 'rb'))

        print("len(scores_shine_pdf_correct)", len(scores_shine_pdf_correct))
        print("len(scores_shine_pdf_incorrect)", len(scores_shine_pdf_incorrect))
        print("len(scores_feature_pdf_correct)", len(scores_feature_pdf_correct))

        min_score_shine_pdf_correct = min(scores_shine_pdf_correct)
        indices_score_shine_pdf_intersection = []
        indices_score_feature_pdf_intersection = []

        for i in range(len(scores_shine_pdf_incorrect)):
            if scores_shine_pdf_incorrect[i] >= min_score_shine_pdf_correct:
                indices_score_shine_pdf_intersection.append(i)

        if len(indices_score_shine_pdf_intersection) > 1:
            selected_features = features[indices_score_shine_pdf_intersection]
            density_estimator = fit_density_estimator(selected_features)
            scores_feature_pdf_incorrect_intersection = density_estimator.score_samples(selected_features)

            # analyzing if we can find better separation between FN and TP now
            min_pdf_feature_correct = min(scores_feature_pdf_correct)

            for i in range(len(scores_feature_pdf_incorrect_intersection)):
                if scores_feature_pdf_incorrect_intersection[i] >= min_pdf_feature_correct:
                    indices_score_feature_pdf_intersection.append(i)

        # let's see if the number of difficult scores to classify as Fn or Tp was improved
        print("before:{}; after:{}\n".format(
            len(indices_score_shine_pdf_intersection),
            len(indices_score_feature_pdf_intersection))
        )


if __name__ == '__main__':
    # multiprocess params
    n_jobs = 4
    # model params
    batch_size = 10
    model = "cnn"  # "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [4]  # [32] # 32 for resnet, 4 for cnn
    additional_transform = None
    adversarial_attack = None

    # dataset params
    id_dataset = "mnist"  # svhn cifar10 mnist
    num_classes = 10
    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    # training set
    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, _, _, pred_train, lab_train = feature_extractor.get_features(dataset_train)
    features = features_train[id_layer_monitored]

    root_path = os.path.join("data_analysis", model, id_dataset)
    filename_format_c = "cls_{}_correct_correct.sav"
    filename_format_i = "cls_{}_correct_incorrect.sav"

    print('data generated from {} on {} dataset'.format(model, id_dataset))
    # arr_data_type = ["feature_similarity", "shine_similarity", "shine_pdf", "feature_pdf"]
    arr_data_type = ["feature_pdf"]
    '''
    # generating data for analysis
    Parallel(n_jobs=n_jobs)(
        delayed(generate_and_save_scores)(num_classes,
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
    '''

    generate_and_save_scores(num_classes,
                             "feature_pdf",
                             x_train,
                             features,
                             lab_train,
                             pred_train,
                             root_path,
                             filename_format_c,
                             filename_format_i)

    '''
    # plotting
    for data_type in arr_data_type:
        print("Loading and generating analysis for:", data_type)
        plot_analysis(num_classes, data_type, root_path, filename_format_c, filename_format_i)
    '''
    # plot_intersection_analysis_feature2shine(num_classes, x_train, root_path, filename_format_c, filename_format_i)
    # plot_intersection_analysis_shine2feature(num_classes, features, root_path, filename_format_c, filename_format_i)
