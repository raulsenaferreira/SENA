# external libs
import numpy as np
from sklearn.metrics import accuracy_score
# internal libs
import utils
import testing_monitors
from dataset import Dataset
from feature_extractor import FeatureExtractor


def main():
    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = \
        feature_extractor.get_features(dataset_train)
    x_test, features_test, logits_test, softmax_test, pred_test, lab_test = \
        feature_extractor.get_features(dataset_test)

    # loading monitors
    monitor_shine = utils.load_monitor(id_dataset, 'shine')
    monitor_oob = utils.load_monitor(id_dataset, 'oob')
    monitor_react = utils.load_monitor(id_dataset, 'react')
    monitor_msp = utils.load_monitor(id_dataset, 'msp')
    monitor_maxlogits = utils.load_monitor(id_dataset, 'maxlogit')
    monitor_energy = utils.load_monitor(id_dataset, 'energy')

    # setting thresholds
    monitor_shine.threshold = id_threshold
    monitor_react.threshold = utils.calculate_threshold(monitor_react, features_train, score_perc=id_threshold)
    monitor_msp.threshold = utils.calculate_threshold(monitor_msp, softmax_train, score_perc=id_threshold)
    monitor_maxlogits.threshold = utils.calculate_threshold(monitor_maxlogits, logits_train, score_perc=id_threshold)
    monitor_energy.threshold = utils.calculate_threshold(monitor_energy, logits_train, score_perc=id_threshold)

    # testing monitors
    monitor_shine.results = []
    monitor_oob.results = []
    monitor_react.results = []
    monitor_msp.results = []
    monitor_maxlogits.results = []
    monitor_energy.results = []

    arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy, monitor_shine]

    # test data (ID)
    m_true_id, arr_monitors = testing_monitors.run(arr_monitors, x_test, pred_test,
                                                   features_test[id_layer_monitored],
                                                   softmax_test, logits_test, lab_test)
    # test data (OOD)
    for transform in additional_transforms:
        dataset_ood = Dataset(ood_dataset, "test", model, additional_transform=transform, batch_size=batch_size)

        x_ood, features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

        m_true_ood, arr_monitors = testing_monitors.run(arr_monitors, x_ood, pred_ood, features_ood[id_layer_monitored],
                                                        softmax_ood, logits_ood, lab_ood)
        # ML model's performance
        id_accuracy = accuracy_score(lab_test, pred_test)
        ood_accuracy = 0
        if id_dataset == ood_dataset:
            ood_accuracy = accuracy_score(lab_ood, pred_ood)
        print("Model accuracy")
        print("ID:  ", id_accuracy)
        print("OOD: ", ood_accuracy)

        # Monitor's performance
        m_true = np.hstack([m_true_id, m_true_ood])
        # saving results
        utils.save_results(arr_monitors, id_dataset, ood_type, transform, m_true)


if __name__ == '__main__':
    ood_type = "distributional_shift"
    additional_transforms = ["blur", "brightness", "pixelization", "shuffle_pixels"]
    # model params
    batch_size = 100
    model = "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]
    id_threshold = 0.9

    # dataset params
    id_dataset = "cifar10"  # "svhn"
    ood_dataset = id_dataset

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)
    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    main()
