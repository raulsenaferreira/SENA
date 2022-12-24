import os
import utils
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, \
    recall_score, confusion_matrix, classification_report
from sklearn.metrics import matthews_corrcoef as mcc


def plotAccuracy(arr, steps, label):
    arr = np.array(arr)
    c = range(len(arr))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arr, 'k')
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(0, steps + 1, 10))
    plt.title(label)
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotF1(arrF1, steps, label):
    arrF1 = np.array(arrF1)
    c = range(len(arrF1))
    fig = plt.figure()
    fig.add_subplot(122)
    ax = plt.axes()
    ax.plot(c, arrF1, 'k')
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    if steps > 10:
        plt.xticks(range(1, steps + 1, 10))
    else:
        plt.xticks(range(1, steps + 1))
    plt.title(label)
    plt.ylabel("F1")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBoxplot(mode, data, labels):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.boxplot(data, labels=labels)
    plt.xticks(rotation=90)

    if mode == 'acc':
        plt.title("Accuracy - Boxplot")
        # plt.xlabel('step (s)')
        plt.ylabel('Accuracy')
    elif mode == 'mcc':
        plt.title('Mathews Correlation Coefficient - Boxplot')
        plt.ylabel("Mathews Correlation Coefficient")
    elif mode == 'f1':
        plt.title('F1 - Boxplot')
        plt.ylabel("F1")

    plt.show()


def plotAccuracyCurves(listOfAccuracies, listOfMethods):
    limit = len(listOfAccuracies[0]) + 1

    for acc in listOfAccuracies:
        acc = np.array(acc)
        c = range(len(acc))
        ax = plt.axes()
        ax.plot(c, acc)

    plt.title("Accuracy curve")
    plt.legend(listOfMethods)
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xticks(range(0, limit, 10))
    plt.ylabel("Accuracy")
    plt.xlabel("Step")
    plt.grid()
    plt.show()


def plotBars(listOfTimes, listOfMethods):
    for l in range(len(listOfTimes)):
        ax = plt.axes()
        ax.bar(l, listOfTimes[l])

    plt.title("Average Accuracy")
    plt.xlabel("Methods")
    plt.ylabel("Accuracy")
    plt.yticks(range(0, 101, 10))
    plt.xticks(range(len(listOfTimes)), listOfMethods)
    plt.xticks(rotation=90)
    plt.grid()
    plt.show()


def finalEvaluation(arrAcc, steps, label):
    print("Average Accuracy: ", np.mean(arrAcc))
    print("Standard Deviation: ", np.std(arrAcc))
    print("Variance: ", np.std(arrAcc) ** 2)
    plotAccuracy(arrAcc, steps, label)


def main(ood_type, ood_config):
    for id_dataset, arr_ood_variant in ood_config.items():
        for ood_variant in arr_ood_variant:
            evaluation_text = ""
            arr_monitor_names = ['shine'] #, 'oob', 'msp', 'maxlogit', 'react', 'energy'
            ground_truth, results = utils.load_results(arr_monitor_names, id_dataset, ood_type, ood_variant)

            for monitor_name in arr_monitor_names:
                tn, fp, fn, tp = confusion_matrix(ground_truth, results[monitor_name]).ravel()
                sg_score = tp/len(ground_truth)
                rh_score = fn/len(ground_truth)
                ac_score = fp/len(ground_truth)
                evaluation_text += "{}" \
                                   "\nMCC: {}" \
                                   "\nBalanced accuracy: {}" \
                                   "\nFPR: {}" \
                                   "\nFNR: {}" \
                                   "\nPrecision: {}" \
                                   "\nRecall: {}" \
                                   "\nF1: {}" \
                                   "\nSG: {}" \
                                   "\nRH: {}" \
                                   "\nAC: {}" \
                                   "\n\n\n".format(monitor_name.upper(),
                                                   mcc(ground_truth, results[monitor_name]),
                                                   balanced_accuracy_score(ground_truth, results[monitor_name]),
                                                   fp / (fp + tn),
                                                   fn / (tp + fn),
                                                   precision_score(ground_truth, results[monitor_name]),
                                                   recall_score(ground_truth, results[monitor_name]),
                                                   f1_score(ground_truth, results[monitor_name]),
                                                   sg_score,
                                                   rh_score,
                                                   ac_score)

                # print(classification_report(ground_truth, results[monitor_name]))

            # saving in a txt file for posterior analysis
            with open(os.path.join("csv", id_dataset, ood_type, ood_variant, "evaluation_output.txt"), "w") as text_file:
                text_file.write(evaluation_text)


if __name__ == '__main__':
    arr_ood_exp_config = {
        "novelty": {"cifar10": ["svhn"]},
        "adversarial": {"cifar10": ["fgsm", "deepfool", "pgd"]},
        "distributional_shift": {"cifar10": ["blur", "brightness", "pixelization", "shuffle_pixels"]}
    }

    for ood_type, ood_config in arr_ood_exp_config.items():
        main(ood_type, ood_config)
