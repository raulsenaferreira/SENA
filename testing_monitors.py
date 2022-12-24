import numpy as np
from tqdm import tqdm


def run(arr_monitors, x_test, pred_test, features_test, softmax_test, logits_test, lab_test):
    m_true = []
    monitor_shine = None
    monitor_oob = None
    monitor_react = None
    monitor_msp = None
    monitor_maxlogits = None
    monitor_energy = None

    for monitor in arr_monitors:
        if monitor is not None:
            if monitor.name == 'shine':
                monitor_shine = monitor
            elif monitor.name == 'oob':
                monitor_oob = monitor
            elif monitor.name == 'react':
                monitor_react = monitor
            elif monitor.name == 'msp':
                monitor_msp = monitor
            elif monitor.name == 'maxlogit':
                monitor_maxlogits = monitor
            elif monitor.name == 'energy':
                monitor_energy = monitor

    for x, pred, feature, softmax, logits, label in tqdm(
            zip(x_test, pred_test, features_test, softmax_test, logits_test, lab_test)):

        if monitor_shine is not None:
            res = monitor_shine.predict(np.array([x]), feature, pred, monitor_shine.threshold_S,
                                        monitor_shine.threshold_HINE)
            monitor_shine.results.append(res)

        elif monitor_oob is not None:
            # oob
            out_box = monitor_oob.predict([feature], [pred])
            monitor_oob.results.append(out_box)

        elif monitor_react is not None:
            # react
            r = monitor_react.predict([feature])
            monitor_react.results.append(r < monitor_react.threshold[pred])

        elif monitor_msp is not None:
            # Max Softmax Probability
            msp = monitor_msp.predict([softmax])
            monitor_msp.results.append(msp < monitor_msp.threshold)

        elif monitor_maxlogits is not None:
            # Max logit
            maxlogits = monitor_maxlogits.predict([logits])
            monitor_maxlogits.results.append(maxlogits < monitor_maxlogits.threshold)

        elif monitor_energy is not None:
            # Energy
            energy = monitor_energy.predict(logits)
            monitor_energy.results.append(energy < monitor_energy.threshold)

        if pred == label:  # monitor does not need to activate
            m_true.append(False)
        else:  # monitor should activate
            m_true.append(True)

    arr_monitors = [monitor_oob, monitor_react, monitor_msp, monitor_maxlogits, monitor_energy, monitor_shine]

    return m_true, arr_monitors
