import numpy as np
from methods import shine
from joblib import Parallel, delayed


def fit_tp(data, cls, labels, predictions, data_type):
    density_estimator_tp = None
    threshold_tp = None

    # all labels from class c
    ind_y_c = np.where(labels == cls)[0]#[:500]
    # all pred as c
    ind_ML_c = np.where(predictions == cls)[0]#[:500]
    # all correct pred as c (TP)
    ind_tp = set(ind_y_c).intersection(ind_ML_c)
    data_tp = data[list(ind_tp)]

    if "feature" == data_type:
        print("fitting TP feature from class {}".format(cls))
        density_estimator_tp = shine.calculate_density(data_tp)
        #self.arr_density_feature_TP.update({cls: density_estimator_tp})
        pdfs = np.exp(density_estimator_tp.score_samples(data_tp))
        threshold_tp = np.mean(pdfs)
        #self.threshold_TP_feature.update({cls: threshold_tp})
    elif "image" == data_type:
        print("fitting TP image from class {}".format(cls))
        matrix_shine = shine.create_matrix_shine(data_tp)
        density_estimator_tp = shine.calculate_density(matrix_shine)
        #self.arr_density_shine_TP.update({cls: density_estimator_tp})
        pdfs = np.exp(density_estimator_tp.score_samples(matrix_shine))
        threshold_tp = np.mean(pdfs)
        #self.threshold_TP_shine.update({cls: threshold_tp})
    else:
        print("incompatible data type!!")

    return density_estimator_tp, threshold_tp


class SHINE_monitor:
    def __init__(self, id_dataset_name, classes_to_monitor=10):
        self.id_dataset_name = id_dataset_name
        self.classes_to_monitor = classes_to_monitor

        # for feature methods
        self.threshold_TP_feature = {}
        self.arr_density_feature_TP = {}

        # for image methods
        self.arr_density_shine_TP = {}
        self.threshold_TP_shine = {}

    def fit(self, X, y, features, ml_preds, parallel=True, n_jobs=4):
        features = np.array(features)

        if parallel:
            res = Parallel(n_jobs=n_jobs)(
                delayed(fit_tp)(X, c, y, ml_preds, "image") for c in range(self.classes_to_monitor)
            )
            for i in range(len(res)):
                self.arr_density_shine_TP.update({i: res[i][0]})
                self.threshold_TP_shine.update({i: res[i][1]})
            # for features is better not to put too much jobs (too much data)
            res = Parallel(n_jobs=int(n_jobs/2))(
                delayed(fit_tp)(features, c, y, ml_preds, "feature") for c in range(self.classes_to_monitor)
            )
            for i in range(len(res)):
                self.arr_density_feature_TP.update({i: res[i][0]})
                self.threshold_TP_feature.update({i: res[i][1]})

        else:
            for c in range(self.classes_to_monitor):
                # Calculating pdf from shine vectors for True Positives
                density_estimator_tp, threshold_tp = fit_tp(X, c, y, ml_preds, "image")
                self.arr_density_shine_TP.update({c: density_estimator_tp})
                self.threshold_TP_shine.update({c: threshold_tp})
                # Calculating pdf from features for True Positives
                density_estimator_tp, threshold_tp = fit_tp(features, c, y, ml_preds, "feature")
                self.threshold_TP_feature.update({c: threshold_tp})
                self.arr_density_feature_TP.update({c: density_estimator_tp})

    def predict(self, X, incoming_feature, pred):
        # SHINE Part 1: comparing feature pdf for TP
        incoming_pdf = np.exp(self.arr_density_feature_TP[pred].score_samples(incoming_feature))[0]

        if incoming_pdf >= self.threshold_TP_feature[pred]:
            return False  # no need to intervene, it's probably an TP
        else:
            # inconclusive, let's do a check using a different source
            matrix_shine = shine.create_matrix_shine(X)
            new_pdf = np.exp(self.arr_density_shine_TP[pred].score_samples(matrix_shine))[0]
            # SHINE Part 2: comparing shine pdf for TP
            if new_pdf <= self.threshold_TP_shine[pred]:
                return True
            else:
                return False
