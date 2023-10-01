import torch.multiprocessing
from matplotlib import pyplot as plt
from scipy import spatial
from Params.params_monitors import *
from methods import shine

torch.multiprocessing.set_sharing_strategy('file_system')


def get_thresholds_from_training_data(scores_TP, scores_FN):
    bp_TP = plt.boxplot(scores_TP, showmeans=True)
    bp_FN = plt.boxplot(scores_FN, showmeans=True)
    # getting thresholds based on quartil 1 for TP and FN
    min_threshold_TP = list(set(bp_TP['boxes'][0].get_ydata()))[0]
    min_threshold_FN = list(set(bp_FN['boxes'][0].get_ydata()))[0]
    # print("min_threshold_TP, min_threshold_FN", min_threshold_TP, min_threshold_FN)
    return min_threshold_TP, min_threshold_FN


def get_thresholds(dataset_name):
    arr_id_threshold = {}
    arr_ood_threshold = {}
    if dataset_name.__eq__("cifar10"):
        arr_id_threshold.update(
            {0: 0.9499, 1: 0.9692, 2: 0.9494, 3: 0.9340, 4: 0.9624, 5: 0.9582, 6: 0.9417, 7: 0.9641,
             8: 0.9695, 9: 0.9695})
        # arr_ood_threshold.update({0: 0.7043, 1: 0.7919, 2: 0.6996, 3: 0.7209, 4:0.7435, 5:0.7924, 6:0.6745,
        # 7:0.7481, 8:0.6934, 9:0.7831})
        arr_ood_threshold.update({0: 0.7, 1: 0.7, 2: 0.6, 3: 0.7, 4: 0.7, 5: 0.7, 6: 0.6, 7: 0.7, 8: 0.6, 9: 0.7})
    elif dataset_name.__eq__("svhn"):
        print("no thresholds available")
        pass

    return arr_id_threshold, arr_ood_threshold


def get_thresholds_2(dataset_name):
    arr_id_threshold = {}
    arr_ood_threshold = {}
    if dataset_name.__eq__("cifar10"):
        arr_id_threshold.update(
            {0: 0.9571, 1: 0.9744, 2: 0.9559, 3: 0.9414, 4: 0.9680, 5: 0.9633, 6: 0.9506, 7: 0.9690,
             8: 0.9751, 9: 0.9740})
        # arr_ood_threshold.update({0: 0.7043, 1: 0.7919, 2: 0.6996, 3: 0.7209, 4:0.7435, 5:0.7924, 6:0.6745,
        # 7:0.7481, 8:0.6934, 9:0.7831})
        arr_ood_threshold.update({0: 0.7, 1: 0.7, 2: 0.6, 3: 0.7, 4: 0.7, 5: 0.7, 6: 0.6, 7: 0.7, 8: 0.6, 9: 0.7})
    elif dataset_name.__eq__("svhn"):
        print("no thresholds available")
        pass

    return arr_id_threshold, arr_ood_threshold


def calculate_feature_similarity_thresholds(arr_f_c_correct, arr_f_c_incorrect):
    scores_TP = []
    scores_FN = []

    for i in range(len(arr_f_c_correct) - 1):
        cosine_similarity = 1 - spatial.distance.cosine(arr_f_c_correct[i], arr_f_c_correct[i + 1])
        scores_TP.append(cosine_similarity)

    for i in arr_f_c_incorrect:
        for c in arr_f_c_correct:
            cosine_similarity = 1 - spatial.distance.cosine(c, i)
            scores_FN.append(cosine_similarity)

    return scores_TP, scores_FN


class SHINE_monitor:
    def __init__(self, id_dataset_name, id_y_train, pred_train, classes_to_monitor=10):
        self.id_dataset_name = id_dataset_name
        self.classes_to_monitor = classes_to_monitor
        self.id_y_train = id_y_train
        self.pred_train = pred_train

        # for S method
        self.min_threshold_TP = {}
        self.min_threshold_FN = {}
        self.arr_repr_feature_TP = {}

        # for HINE methods
        self.arr_density_img = {}
        self.scores_FN_HINE = {}

    def calculate_feature_similarity(self, pred, incoming_feature):
        f_c_correct = self.arr_repr_feature_TP[pred]

        scores = []
        for c in f_c_correct:
            cosine_similarity = 1 - spatial.distance.cosine(c, incoming_feature)
            scores.append(cosine_similarity)

        bp_incoming_X = plt.boxplot(scores, showmeans=True)
        # getting thresholds based on quartil 1 for TP and FN
        min_acc_threshold = list(set(bp_incoming_X['boxes'][0].get_ydata()))[0]

        return min_acc_threshold

    def calculate_img_stats(self, X, pred, threshold_SMimg):
        scores = self.scores_FN_HINE[pred]
        len_scores = len(scores)
        sorted_scores = sorted(scores)
        id_threshold = int(len_scores * threshold_SMimg)
        threshold = sorted_scores[id_threshold]  # -id_threshold if threshold_SMimg = 1 - 0.9

        matrix_shine = shine.create_matrix_shine(X)
        logprob = self.arr_density_img[pred].score_samples(matrix_shine)

        return np.exp(logprob)[0] <= threshold, np.exp(logprob)

    def fit_by_class(self, X, y, features, ml_predictions):
        features = np.array(features)

        for c in range(self.classes_to_monitor):
            # all labels from class c
            ind_y_c = np.where(y == c)[0]

            # SHINE Part 1: calculating scores for S
            # Correct predictions
            ind_ML_c = np.where(ml_predictions == c)[0]
            self.arr_repr_feature_TP.update({c: features[list(ind_ML_c)]})
            # Incorrect predictions
            ind_incorrect_ML = set(ind_y_c).symmetric_difference(ind_ML_c)
            # self.arr_repr_feature_FN.update({c: features[list(ind_incorrect_ML)]})

            scores_TP, scores_FN = calculate_feature_similarity_thresholds(features[list(ind_ML_c)],
                                                                           features[list(ind_incorrect_ML)])
            # self.scores_TP_S.update({c: scores_TP})  # NEW
            # self.scores_FN_S.update({c: scores_FN})  # NEW

            # min_threshold_TP, min_threshold_FN = get_thresholds_from_training_data(scores_TP, scores_FN)  # NEW
            # self.min_threshold_TP.update({c: min_threshold_TP})  # NEW
            # self.min_threshold_FN.update({c: min_threshold_FN})  # NEW
            min_threshold_TP, min_threshold_FN = get_thresholds_2("cifar10")
            self.min_threshold_TP.update({c: min_threshold_TP[c]})
            self.min_threshold_FN.update({c: min_threshold_FN[c]})

            # SHINE Part 2: calculating scores for HINE
            self.arr_density_img.update({c: None})
            self.scores_FN_HINE.update({c: None})
            X_c_incorrect = X[list(ind_incorrect_ML)]

            if len(X_c_incorrect) > 1:
                matrix_shine = shine.create_matrix_shine(X_c_incorrect)
                density_estimator = shine.calculate_density(matrix_shine)  # , False, band_list[c])
                self.arr_density_img.update({c: density_estimator})
                pdfs = np.exp(density_estimator.score_samples(matrix_shine))
                self.scores_FN_HINE.update({c: pdfs})

    def predict(self, X, incoming_feature, pred, threshold_HINE):
        # SHINE Part 1: comparing thresholds for S
        min_acc_threshold = self.calculate_feature_similarity(pred, incoming_feature)  # NEW

        if min_acc_threshold >= self.min_threshold_TP[pred]:  # NEW  # arr_tp_threshold[pred]:
            return False  # it is probably an TP (correct pred)
        elif min_acc_threshold <= self.min_threshold_FN[pred]:  # NEW  # arr_fn_threshold[pred]:
            return True  # it is probably an FN (incorrect pred)
        # SHINE part 2: it is inconclusive if the prediction is TP or FN: comparing threshold for HINE
        else:
            if self.scores_FN_HINE[pred] is not None:
                monitor_pred, pdf = self.calculate_img_stats(X, pred, threshold_HINE)
                return monitor_pred
            else:
                # there weren't incorrect instances during training for this class, so you can trust the ML model
                return False
