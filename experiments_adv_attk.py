# external libs
import numpy as np
from sklearn.metrics import matthews_corrcoef as mcc
# internal libs
import utils
import testing_monitors
from dataset import Dataset
from feature_extractor import FeatureExtractor


def main():
    # .threshold = utils.calculate_threshold(monitor_react, features_train, score_perc=id_threshold)
    ood_type = "adversarial"
    adversarial_attacks = ["fgsm", "deepfool", "pgd"]
    # model params
    batch_size = 100
    model = "resnet"  # "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]  # 32

    # dataset params
    id_dataset = "svhn"  # "svhn" "cifar10"
    ood_dataset = id_dataset

    # thresholds collected considering better mcc
    threshold_cifar10 = {'react': 0.661, 'msp': 0.863, 'maxlogit': 0.47, 'energy': 0.473}
    threshold_svhn = {'react': 0.565, 'msp': 0.609, 'maxlogit': 0.431, 'energy': 0.489}
    threshold_mnist = {'react': 0.999, 'msp': 0.744, 'maxlogit': 0.1, 'energy': 0.592}
    threshold_fashion_mnist = {'react': 0.999, 'msp': 0.743, 'maxlogit': 0.1, 'energy': 0.313}

    id_thresholds = {'cifar10': threshold_cifar10, 'svhn': threshold_svhn,
                     "mnist": threshold_mnist, "fashion_mnist": threshold_fashion_mnist}

    dataset_test = Dataset(id_dataset, "test", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)

    x_test, features_test, logits_test, softmax_test, pred_test, lab_test = \
        feature_extractor.get_features(dataset_test)

    # loading monitors
    monitor_oob = utils.load_monitor(id_dataset, 'oob')
    monitor_react = utils.load_monitor(id_dataset, 'react')
    monitor_msp = utils.load_monitor(id_dataset, 'msp')
    monitor_maxlogits = utils.load_monitor(id_dataset, 'maxlogit')
    monitor_energy = utils.load_monitor(id_dataset, 'energy')

    # setting thresholds (for more details about how thresholds were chosen, see file "find_best_thresholds.py")
    monitor_react.threshold = id_thresholds[id_dataset]['react']
    monitor_msp.threshold = id_thresholds[id_dataset]['msp']
    monitor_maxlogits.threshold = id_thresholds[id_dataset]['maxlogit']
    monitor_energy.threshold = id_thresholds[id_dataset]['energy']

    # test data (ID)
    # m_true_id, arr_monitors = testing_monitors.run(arr_monitors, x_test, pred_test,
    #                                               features_test[id_layer_monitored],
    #                                               softmax_test, logits_test, lab_test)

    m_true_id = pred_test != lab_test

    for attack in adversarial_attacks:
        print("Adv attack experiments with {} as ID and {} as OOD attack".format(id_dataset, attack))
        monitor_oob.results = []
        monitor_react.results = []
        monitor_msp.results = []
        monitor_maxlogits.results = []
        monitor_energy.results = []

        # oob
        monitor_oob.results = monitor_oob.predict(features_test[id_layer_monitored], pred_test)
        # react
        scores_react = monitor_react.predict(features_test[id_layer_monitored])
        monitor_react.results = scores_react < monitor_react.threshold
        # msp
        scores_msp = monitor_msp.predict(softmax_test)
        monitor_msp.results = scores_msp < monitor_msp.threshold
        # max logits
        scores_maxlogits = monitor_maxlogits.predict(logits_test)
        scores_maxlogits *= (1 / np.max(scores_maxlogits))
        monitor_maxlogits.results = scores_maxlogits < monitor_maxlogits.threshold
        # Energy
        scores_energy = monitor_energy.predict(logits_test)
        scores_energy *= (1 / np.max(scores_energy))
        monitor_energy.results = scores_energy < monitor_energy.threshold

        dataset_ood = Dataset(ood_dataset, "test", model, adversarial_attack=attack, batch_size=batch_size)

        x_ood, features_ood, logits_ood, softmax_ood, pred_ood, lab_ood = feature_extractor.get_features(dataset_ood)

        # saving ML model's performance (Baseline)
        utils.save_baseline_results(ood_type, attack, id_dataset, ood_dataset,
                                    pred_test, lab_test, pred_ood, lab_ood)

        # ground truth
        m_true_ood = pred_ood != lab_ood
        m_true = np.hstack([m_true_id, m_true_ood])

        # oob
        ood_results = monitor_oob.predict(features_ood[id_layer_monitored], pred_ood)
        monitor_oob.results = np.hstack([monitor_oob.results, ood_results])
        # react
        scores_react = monitor_react.predict(features_ood[id_layer_monitored])
        ood_results = scores_react < monitor_react.threshold
        monitor_react.results = np.hstack([monitor_react.results, ood_results])
        # msp
        scores_msp = monitor_msp.predict(softmax_ood)
        ood_results = scores_msp < monitor_msp.threshold
        monitor_msp.results = np.hstack([monitor_msp.results, ood_results])
        # max logits
        scores_maxlogits = monitor_maxlogits.predict(logits_ood)
        scores_maxlogits *= (1 / np.max(scores_maxlogits))
        ood_results = scores_maxlogits < monitor_maxlogits.threshold
        monitor_maxlogits.results = np.hstack([monitor_maxlogits.results, ood_results])
        # Energy
        scores_energy = monitor_energy.predict(logits_ood)
        scores_energy *= (1 / np.max(scores_energy))
        ood_results = scores_energy < monitor_energy.threshold
        monitor_energy.results = np.hstack([monitor_energy.results, ood_results])

        arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy]
        # saving results
        utils.save_results(arr_monitors, id_dataset, ood_type, attack, m_true)


if __name__ == '__main__':
    main()
