# internal libs
import utils
from dataset import Dataset
from feature_extractor import FeatureExtractor
from monitors_internals import MaxSoftmaxProbabilityMonitor, MaxLogitMonitor, EnergyMonitor, ReActMonitor


def main():
    # model params
    batch_size = 100
    model = "resnet"

    # monitor params
    layer_relu_ids = [32]

    # dataset params
    id_dataset = "cifar10"

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(dataset_train)

    # building react monitor
    monitor_react = ReActMonitor(quantile_value=0.99, mode='msp')
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

    threshold_react = utils.calculate_threshold(monitor_react, features_train, score_perc=0.93)
    threshold_msp = utils.calculate_threshold(monitor_msp, softmax_train, score_perc=0.93)
    threshold_maxlogits = utils.calculate_threshold(monitor_maxlogits, logits_train, score_perc=0.93)
    threshold_energy = utils.calculate_threshold(monitor_energy, logits_train, score_perc=0.93)

    print('threshold_react', threshold_react)
    print('threshold_msp', threshold_msp)
    print('threshold_maxlogits', threshold_maxlogits)
    print('threshold_energy', threshold_energy)


if __name__ == '__main__':
    main()
