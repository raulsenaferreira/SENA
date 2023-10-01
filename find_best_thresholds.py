import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef as mcc
from scipy.special import logsumexp, softmax
# internal libs
import utils
from dataset import Dataset
from feature_extractor import FeatureExtractor
from monitors_internals import MaxSoftmaxProbabilityMonitor, MaxLogitMonitor, EnergyMonitor, \
    ReActMonitor, MahalanobisMonitor


def get_val_threshold(scores, score_threshold):
    arr_scores = np.array(scores)  # (arr_scores)
    arr_scores.sort()
    limit = int(len(arr_scores) * score_threshold)

    return arr_scores[limit]


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


def main():
    # model params
    batch_size = 100
    model = "cnn"  # "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [4]  # [32]

    # dataset params
    id_dataset = "fashion_mnist"  # "mnist" "cifar10" "svhn"

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = \
        feature_extractor.get_features(dataset_train)

    # building react monitor
    monitor_react = ReActMonitor(mode='msp')
    monitor_react.fit(feature_extractor, features_train)
    monitor_react.name = 'react'
    # building softmax monitor
    monitor_msp = MaxSoftmaxProbabilityMonitor()
    monitor_msp.fit()
    monitor_msp.name = 'msp'
    # building max logit monitor
    monitor_maxlogits = MaxLogitMonitor()
    monitor_maxlogits.fit()
    monitor_maxlogits.name = 'maxlogit'
    # building Energy monitor
    monitor_energy = EnergyMonitor(temperature=1)
    monitor_energy.fit()
    monitor_energy.name = 'energy'
    # building Mahalanobis monitor
    monitor_mahalanobis = MahalanobisMonitor(id_dataset, model, id_layer_monitored)
    monitor_mahalanobis.fit(features_train[id_layer_monitored], lab_train)
    monitor_mahalanobis.name = 'mahalanobis'

    #for cls in range(10):
    best_f1_react = 0
    best_t_react = 0

    best_f1_msp = 0
    best_t_msp = 0

    best_f1_maxlogits = 0
    best_t_maxlogits = 0

    best_f1_energy = 0
    best_t_energy = 0

    best_f1_mahalanobis = 0
    best_t_mahalanobis = 0

    # all labels from class c
    #ind_cls = np.where(lab_train == cls)[0]
    ground_truth = pred_train != lab_train  # monitors should react to all wrong ML predictions

    features_train = np.array(features_train)
    arr_threshold = np.arange(0.1, 1, 0.001).tolist()

    for threshold in arr_threshold:
        threshold = round(threshold, 3)
        #'''
        scores_react = monitor_react.predict(features_train[id_layer_monitored])
        res_react = scores_react < threshold
        f1_react = mcc(ground_truth, res_react)

        scores_msp = monitor_msp.predict(softmax_train)
        res_msp = scores_msp < threshold
        f1_msp = mcc(ground_truth, res_msp)

        scores_maxlogits = monitor_maxlogits.predict(logits_train)
        scores_maxlogits *= (1/np.max(scores_maxlogits))
        res_maxlogits = scores_maxlogits < threshold
        f1_maxlogits = mcc(ground_truth, res_maxlogits)

        scores_energy = monitor_energy.predict(logits_train)
        scores_energy *= (1/np.max(scores_energy))
        res_energy = scores_energy < threshold
        f1_energy = mcc(ground_truth, res_energy)
        '''
        scores_mahalanobis = monitor_mahalanobis.predict(features_train[id_layer_monitored], pred_train)
        scores_mahalanobis *= (1 / np.max(scores_mahalanobis))
        res_mahalanobis = scores_mahalanobis < threshold
        f1_mahalanobis = f1_score(ground_truth, res_mahalanobis)
        '''
        if f1_react > best_f1_react:
            best_f1_react = f1_react
            best_t_react = threshold

        if f1_msp > best_f1_msp:
            best_f1_msp = f1_msp
            best_t_msp = threshold

        if f1_maxlogits > best_f1_maxlogits:
            best_f1_maxlogits = f1_maxlogits
            best_t_maxlogits = threshold

        if f1_energy > best_f1_energy:
            best_f1_energy = f1_energy
            best_t_energy = threshold
        '''
        if f1_mahalanobis > best_f1_mahalanobis:
            best_f1_mahalanobis = f1_mahalanobis
            best_t_mahalanobis = threshold
        '''

    print('best threshold react {} based on F1={}'.format(best_t_react, best_f1_react))
    print('best threshold msp {} based on F1={}'.format(best_t_msp, best_f1_msp))
    print('best threshold maxlogit {} based on F1={}'.format(best_t_maxlogits, best_f1_maxlogits))
    print('best threshold energy {} based on F1={}'.format(best_t_energy, best_f1_energy))

    #print('best threshold mahalanobis {} based on F1={}'.format(best_t_mahalanobis, best_f1_mahalanobis))


if __name__ == '__main__':
    main()
    '''
    thresholds for better F1
    
    Thresholds for cifar10:
    best threshold react 0.661 based on F1=0.42424242424242425
    best threshold msp 0.715 based on F1=0.4123711340206186
    best threshold maxlogit 0.39 based on F1=0.26900584795321636
    best threshold energy 0.417 based on F1=0.2262210796915167
    
    Thresholds for svhn:
    best threshold react 0.565 based on F1=0.5714285714285713
    best threshold msp 0.644 based on F1=0.6
    best threshold maxlogit 0.411 based on F1=0.3181818181818182
    best threshold energy 0.428 based on F1=0.2553191489361702
    
    thresholds for better MCC
    
    Thresholds for cifar10:
    best threshold react 0.661 based on MCC=0.42708975544621247
    best threshold msp 0.863 based on MCC=0.41808579402866647
    best threshold maxlogit 0.47 based on MCC=0.30511892103221105
    best threshold energy 0.473 based on MCC=0.27454384594800896
    
    Thresholds for svhn:
    best threshold react 0.565 based on MCC=0.5720163728330104
    best threshold msp 0.609 based on MCC=0.6154179643435131
    best threshold maxlogit 0.431 based on MCC=0.36767903412162734
    best threshold energy 0.489 based on MCC=0.3158578265992241
    
    Thresholds for mnist:
    best threshold react 0.999 based on F1=0.09063451415027396
    best threshold msp 0.744 based on F1=0.4762672783101345
    best threshold maxlogit 0.1 based on F1=0.03775950024564679
    best threshold energy 0.592 based on F1=0.002474295460761397
    
    Thresholds for fashion mnist
    best threshold react 0.999 based on F1=0.23359860424645212
    best threshold msp 0.743 based on F1=0.4665696454932824
    best threshold maxlogit 0.1 based on F1=0.10852159280727668
    best threshold energy 0.313 based on F1=0.007215805316179716

    '''