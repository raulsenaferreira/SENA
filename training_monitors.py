# internal libs
import utils
from dataset import Dataset
from feature_extractor import FeatureExtractor
from monitors_internals import OutsideTheBoxMonitor, MaxSoftmaxProbabilityMonitor, MaxLogitMonitor, EnergyMonitor, \
    ReActMonitor
from monitors_input import SHINE_monitor2


def main():
    # model params
    batch_size = 100
    model = "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [32]

    # dataset params
    id_dataset = "cifar10" #"svhn"

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
        dataset_train)

    # building oob monitor
    monitor_oob = OutsideTheBoxMonitor(n_clusters=3)
    monitor_oob.fit(features_train[id_layer_monitored], lab_train)
    # building react monitor
    monitor_react = ReActMonitor(quantile_value=0.99, mode='msp')
    monitor_react.fit(feature_extractor, features_train[id_layer_monitored])
    # building softmax monitor
    monitor_msp = MaxSoftmaxProbabilityMonitor()
    monitor_msp.fit()
    # building max logit monitor
    monitor_maxlogits = MaxLogitMonitor()
    monitor_maxlogits.fit()
    # building Energy monitor
    monitor_energy = EnergyMonitor(temperature=1)
    monitor_energy.fit()
    # building SHINE monitor
    monitor_shine = SHINE_monitor2(id_dataset, lab_train, pred_train)
    monitor_shine.fit_by_class(x_train, lab_train, features_train[id_layer_monitored], pred_train)
    print('number of monitors', len(monitor_shine.arr_density_img))

    # saving monitors
    monitor_shine.name = 'shine'
    monitor_oob.name = 'oob'
    monitor_energy.name = 'energy'
    monitor_maxlogits.name = 'maxlogit'
    monitor_msp.name = 'msp'
    monitor_react.name = 'react'

    arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy, monitor_shine]
    utils.save_monitors(arr_monitors, id_dataset)


if __name__ == '__main__':
    main()
