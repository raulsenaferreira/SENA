import torch.multiprocessing
from matplotlib import pyplot as plt
from scipy import spatial
from Params.params_monitors import *
from methods import shine

torch.multiprocessing.set_sharing_strategy('file_system')


def get_thresholds_from_training_data(scores_fn, threshold_type="avg"):
    bp_FN = plt.boxplot(scores_fn, showmeans=True)

    if threshold_type == "avg":
        mean = list(set(bp_FN['means'][0].get_ydata()))
        return mean
    elif threshold_type == "q1":
        boxes_cor = list(set(bp_FN['boxes'][0].get_ydata()))
        return min(boxes_cor)
    elif threshold_type == "q3":
        boxes_cor = list(set(bp_FN['boxes'][0].get_ydata()))
        return max(boxes_cor)


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
    scores_FN = []

    for i in arr_f_c_incorrect:
        for c in arr_f_c_correct:
            cosine_similarity = 1 - spatial.distance.cosine(c, i)
            scores_FN.append(cosine_similarity)

    return scores_FN


class SHINE_monitor:
    def __init__(self, id_dataset_name, id_y_train, pred_train, classes_to_monitor=10):
        self.id_dataset_name = id_dataset_name
        self.classes_to_monitor = classes_to_monitor
        self.id_y_train = id_y_train
        self.pred_train = pred_train

        # for S method
        self.min_threshold_TP = {}
        self.min_threshold_FN = {}
        self.scores_TP_feature = {}
        self.arr_density_feature_TP = {}

        # for HINE methods
        self.arr_repr_FN = {}

    def calculate_is_tp(self, pred, incoming_feature, scores, threshold_pdf_tp):
        logprob_feature = self.arr_density_feature_TP[pred].score_samples(incoming_feature)
        pdf_incoming_feature = np.exp(logprob_feature)[0]
        len_scores = len(scores)
        sorted_scores = sorted(scores)
        calculated_threshold_ind = int(len_scores * threshold_pdf_tp)
        threshold = sorted_scores[calculated_threshold_ind]  # -calculated_threshold if threshold_pdf_tp = 1 - 0.9

        return pdf_incoming_feature >= threshold

    def calculate_is_fn(self, pred, matrix_shine, threshold_type="avg"):
        scores = []
        for i in self.arr_repr_FN[pred]:
            for c in matrix_shine:
                cosine_similarity = 1 - spatial.distance.cosine(c, i)
                scores.append(cosine_similarity)

        bp_FN = plt.boxplot(scores, showmeans=True)

        if threshold_type == "avg":
            mean = list(set(bp_FN['means'][0].get_ydata()))
            return self.min_threshold_FN[pred] <= mean

    def fit_by_class(self, X, y, features, ml_predictions):
        features = np.array(features)

        for c in range(self.classes_to_monitor):
            # all labels from class c
            ind_y_c = np.where(y == c)[0]

            # SHINE Part 1: calculating scores for S
            # Correct predictions
            ind_ML_correct = np.where(ml_predictions == c)[0]
            print("fitting density estimation for class {}".format(c))
            density_estimator_feature_tp = shine.calculate_density(features[list(ind_ML_correct)])
            self.arr_density_feature_TP.update({c: density_estimator_feature_tp})
            pdfs = np.exp(self.arr_density_feature_TP[c].score_samples(features[list(ind_ML_correct)]))
            self.scores_TP_feature.update({c: pdfs})

            # SHINE Part 2: calculating scores for HINE
            # Incorrect predictions
            ind_incorrect_ML = set(ind_y_c).symmetric_difference(ind_ML_correct)
            self.arr_repr_FN.update({c: None})

            if len(ind_incorrect_ML) > 1:
                X_c_incorrect = X[list(ind_incorrect_ML)]
                matrix_shine_incorrect = shine.create_matrix_shine(X_c_incorrect)
                self.arr_repr_FN.update({c: matrix_shine_incorrect})

                X_c_correct = X[list(ind_ML_correct)]
                matrix_shine_correct = shine.create_matrix_shine(X_c_correct)

                scores_FN = calculate_feature_similarity_thresholds(matrix_shine_correct, matrix_shine_incorrect)

                min_threshold_FN = get_thresholds_from_training_data(scores_FN, threshold_type="avg")  # NEW
                self.min_threshold_FN.update({c: min_threshold_FN})  # NEW
                # min_threshold_TP, min_threshold_FN = get_thresholds_2("cifar10")
                # self.min_threshold_FN.update({c: min_threshold_FN})

    def predict(self, X, incoming_feature, pred, threshold_pdf_tp):
        # SHINE Part 1: comparing thresholds for S
        scores = self.scores_TP_feature[pred]
        is_tp = self.calculate_is_tp(pred, incoming_feature, scores, threshold_pdf_tp)
        if is_tp:
            return False  # it is probably an TP (correct pred)
        else:
            # SHINE part 2: it is inconclusive if the prediction is TP or FN: comparing threshold for HINE
            if self.arr_repr_FN[pred] is not None:
                matrix_shine = shine.create_matrix_shine(X)
                is_fn = self.calculate_is_fn(pred, matrix_shine)
                return is_fn
            else:
                # there weren't incorrect instances during training for this class, so you can trust the ML model
                return False
