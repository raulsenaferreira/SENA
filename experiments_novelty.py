# external libs
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
# internal libs
import utils
from dataset import Dataset
from feature_extractor import FeatureExtractor


def main():
    # model params
    batch_size = 100
    model = "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]
    id_threshold = 0.9

    # dataset params
    id_dataset = "cifar10" #"svhn" #
    arr_ood_dataset = ["svhn"] #["cifar10"]#

    # experiment params
    ood_type = "novelty"

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    for ood_dataset in arr_ood_dataset:
        variant = ood_dataset
        dataset_ood = Dataset(ood_dataset, "test", model, batch_size=batch_size)

        feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
        x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
            dataset_train)
        x_test, features_test, logits_test, softmax_test, pred_test, lab_test = feature_extractor.get_features(dataset_test)
        x_ood, features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

        # accuracy of the ML model
        id_accuracy = accuracy_score(lab_test, pred_test)
        ood_accuracy = 0
        if id_dataset == ood_dataset:
            ood_accuracy = accuracy_score(lab_ood, pred_ood)
        print("Model accuracy")
        print("ID:  ", id_accuracy)
        print("OOD: ", ood_accuracy)

        # loading monitors
        monitor_shine = utils.load_monitor(id_dataset, 'shine')
        monitor_oob = utils.load_monitor(id_dataset, 'oob')
        monitor_react = utils.load_monitor(id_dataset, 'react')
        monitor_msp = utils.load_monitor(id_dataset, 'msp')
        monitor_maxlogits = utils.load_monitor(id_dataset, 'maxlogit')
        monitor_energy = utils.load_monitor(id_dataset, 'energy')

        # setting thresholds
        monitor_shine.threshold_HINE = 1 - id_threshold
        monitor_shine.threshold_S = id_threshold
        monitor_react.threshold = utils.calculate_threshold(monitor_react, features_train, score_perc=id_threshold)
        monitor_msp.threshold = utils.calculate_threshold(monitor_msp, softmax_train, score_perc=id_threshold)
        monitor_maxlogits.threshold = utils.calculate_threshold(monitor_maxlogits, logits_train, score_perc=id_threshold)
        monitor_energy.threshold = utils.calculate_threshold(monitor_energy, logits_train, score_perc=id_threshold)

        # testing monitors
        m_true = []
        monitor_shine.results = []
        monitor_oob.results = []
        monitor_react.results = []
        monitor_msp.results = []
        monitor_maxlogits.results = []
        monitor_energy.results = []

        # test data (ID)
        for x, pred, feature, softmax, logits, label in tqdm(
                zip(x_test, pred_test, features_test[id_layer_monitored],
                    softmax_test, logits_test, lab_test)):

            res = monitor_shine.predict(np.array([x]), feature, pred, monitor_shine.threshold_S,
                                        monitor_shine.threshold_HINE)
            monitor_shine.results.append(res)

            # oob
            out_box = monitor_oob.predict([feature], [pred])
            monitor_oob.results.append(out_box)

            # react
            r = monitor_react.predict([feature])
            monitor_react.results.append(r < monitor_react.threshold[pred])

            # Max Softmax Probability
            msp = monitor_msp.predict([softmax])
            monitor_msp.results.append(msp < monitor_msp.threshold)

            # Max logit
            maxlogits = monitor_maxlogits.predict([logits])
            monitor_maxlogits.results.append(maxlogits < monitor_maxlogits.threshold)

            # Energy
            energy = monitor_energy.predict(logits)
            monitor_energy.results.append(energy < monitor_energy.threshold)

            if pred == label:  # monitor does not need to activate
                m_true.append(False)
            else:  # monitor should activate
                m_true.append(True)

        # novelty data
        for x, feature, softmax, logits, pred in tqdm(
                zip(x_ood, features_ood[id_layer_monitored], softmax_ood,
                    logits_ood, pred_ood)):

            res = monitor_shine.predict(np.array([x]), feature, pred, monitor_shine.threshold_S,
                                        monitor_shine.threshold_HINE)
            monitor_shine.results.append(res)

            # oob
            out_box = monitor_oob.predict([feature], [pred])
            monitor_oob.results.append(out_box)

            # react
            r = monitor_react.predict([feature])
            monitor_react.results.append(r < monitor_react.threshold[pred])

            # Max Softmax Probability
            msp = monitor_msp.predict([softmax])
            monitor_msp.results.append(msp < monitor_msp.threshold)

            # Max logit
            maxlogits = monitor_maxlogits.predict([logits])
            monitor_maxlogits.results.append(maxlogits < monitor_maxlogits.threshold)

            # Energy
            energy = monitor_energy.predict(logits)
            monitor_energy.results.append(energy < monitor_energy.threshold)

            m_true.append(True)  # monitor should always react to novel classes

        # saving results
        arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy, monitor_shine]

        utils.save_results(arr_monitors, id_dataset, ood_type, variant, m_true)


if __name__ == '__main__':
    main()
