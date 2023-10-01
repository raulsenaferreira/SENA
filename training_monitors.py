# internal libs
import utils
from dataset import Dataset
from feature_extractor import FeatureExtractor
from monitors_internals import OutsideTheBoxMonitor, MaxSoftmaxProbabilityMonitor, MaxLogitMonitor, EnergyMonitor, \
    ReActMonitor
from monitors_input_4 import SHINE_monitor


def main():
    # model params
    batch_size = 100
    model = "cnn"  # "resnet"

    # monitor params
    id_layer_monitored = -1
    layer_relu_ids = [4]  # [32]  [4]

    # dataset params
    id_dataset = "fashion_mnist"  # "svhn" "cifar10" "mnist" "fashion_mnist"

    dataset_train = Dataset(id_dataset, "train", model, batch_size=batch_size)

    feature_extractor = FeatureExtractor(model, id_dataset, layer_relu_ids)
    x_train, features_train, logits_train, softmax_train, pred_train, lab_train = feature_extractor.get_features(
        dataset_train)

    # building oob monitor
    monitor_oob = OutsideTheBoxMonitor(n_clusters=3)
    monitor_oob.fit(features_train[id_layer_monitored], lab_train)
    # building react monitor
    monitor_react = ReActMonitor(mode='msp')
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
    #monitor_shine = SHINE_monitor(id_dataset)
    #monitor_shine.fit(x_train, lab_train, features_train[id_layer_monitored], pred_train)
    #print('number of pdf feature monitors', len(monitor_shine.arr_density_feature_TP))
    #print('number of pdf shine monitors', len(monitor_shine.arr_density_shine_TP))

    # saving monitors
    #monitor_shine.name = 'shine'
    monitor_oob.name = 'oob'
    monitor_energy.name = 'energy'
    monitor_maxlogits.name = 'maxlogit'
    monitor_msp.name = 'msp'
    monitor_react.name = 'react'

    arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy]
    utils.save_monitors(arr_monitors, id_dataset)


if __name__ == '__main__':
    main()
