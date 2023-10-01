import os
import utils
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from Orange import evaluation
from scipy import stats
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
            arr_monitor_names = ['oob', 'msp', 'maxlogit', 'react', 'energy']  #
            ground_truth, results = utils.load_results(arr_monitor_names, id_dataset, ood_type, ood_variant)

            for monitor_name in arr_monitor_names:
                try:
                    tn, fp, fn, tp = confusion_matrix(ground_truth, results[monitor_name]).ravel()
                    sg_score = tp / len(ground_truth)
                    rh_score = fn / len(ground_truth)
                    ac_score = fp / len(ground_truth)
                    evaluation_text += "{}" \
                                       "\nMCC: {}" \
                                       "\nBalanced accuracy: {}" \
                                       "\nFPR: {}" \
                                       "\nFNR: {}" \
                                       "\nPrecision: {}" \
                                       "\nRecall: {}" \
                                       "\nMacro F1: {}" \
                                       "\nSG: {}" \
                                       "\nRH: {}" \
                                       "\nAC: {}" \
                                       "\n\n\n".format(monitor_name.upper(),
                                                       mcc(ground_truth, results[monitor_name]),
                                                       balanced_accuracy_score(ground_truth, results[monitor_name]),
                                                       fp / (fp + tn),
                                                       fn / (tp + fn),
                                                       precision_score(ground_truth, results[monitor_name], average='macro'),
                                                       recall_score(ground_truth, results[monitor_name], average='macro'),
                                                       f1_score(ground_truth, results[monitor_name], average='macro'),
                                                       sg_score,
                                                       rh_score,
                                                       ac_score)

                    # print(classification_report(ground_truth, results[monitor_name]))
                except:
                    print('Exception occurred for {} result'.format(ood_variant))
            # saving in a txt file for posterior analysis
            with open(os.path.join("csv", id_dataset, ood_type, ood_variant, "evaluation_output.txt"),
                      "w") as text_file:
                text_file.write(evaluation_text)


def load_val(monitor_name, ood_type, ood_config):
    arr_mcc = []
    arr_bal_acc = []
    arr_FPR = []
    arr_FNR = []
    arr_precision = []
    arr_recall = []
    arr_macro_F1 = []
    arr_sg = []
    arr_rh = []
    arr_ac = []

    for id_dataset, arr_ood_variant in ood_config.items():

        for ood_variant in arr_ood_variant:
            ground_truth, results = None, None
            arr_monitor_names = ['baseline', 'oob', 'msp', 'maxlogit', 'react', 'energy']
            if monitor_name == 'sena':
                ground_truth, results = utils.load_sena_results(id_dataset, ood_type, ood_variant)
            elif monitor_name == 'baseline':
                # ground_truth, results = utils.load_baseline_results(id_dataset, ood_type, ood_variant)
                _, unpickled_data = utils.load_results(arr_monitor_names, id_dataset, ood_type, ood_variant)
                ground_truth = unpickled_data[monitor_name][0]
                results = unpickled_data[monitor_name][1]
            else:
                ground_truth, results = utils.load_results(arr_monitor_names, id_dataset, ood_type, ood_variant)
                results = results[monitor_name]

            if monitor_name == 'baseline':
                CF = confusion_matrix(ground_truth, results)
                fp = CF.sum(axis=0) - np.diag(CF)
                fn = CF.sum(axis=1) - np.diag(CF)
                tp = np.diag(CF)
                tn = CF.sum() - (fp + fn + tp)

            else:
                tn, fp, fn, tp = confusion_matrix(ground_truth, results).ravel()

            sg_score = tp / len(ground_truth)
            rh_score = fn / len(ground_truth)
            ac_score = fp / len(ground_truth)
            arr_mcc.append(mcc(ground_truth, results))
            arr_bal_acc.append(balanced_accuracy_score(ground_truth, results))
            arr_FPR.append(fp / (fp + tn))
            arr_FNR.append(fn / (tp + fn))
            pr = precision_score(ground_truth, results, average='macro')
            arr_precision.append(pr)
            re = recall_score(ground_truth, results, average='macro')
            arr_recall.append(re)
            F1 = f1_score(ground_truth, results, average='macro')
            arr_macro_F1.append(F1)
            arr_sg.append(sg_score)
            arr_rh.append(rh_score)
            arr_ac.append(ac_score)

    dict_results = {'mcc': arr_mcc, 'bal_acc': arr_bal_acc, 'fpr': arr_FPR, 'fnr': arr_FNR,
                    'precision': arr_precision, 'recall': arr_recall, 'macro_f1': arr_macro_F1,
                    'sg': arr_sg, 'rh': arr_rh, 'ac': arr_ac}
    return dict_results


def calculate_ranks(all_results):
    from scipy.stats import rankdata
    all_results = np.array(all_results)
    # trick below for reversed rank
    all_results = all_results * -1
    arr_ranks = []
    len_res = len(all_results[0])

    for i in range(len_res):
        # temp = all_results[:, i].argsort()[::-1]
        ranks = rankdata(all_results[:, i], method='min')
        # ranks = np.empty_like(temp)
        # ranks[temp] = np.arange(len(all_results[:, i])) + 1
        arr_ranks.append(ranks)

    return arr_ranks


def plot_statistical_test_heatmap(title, labels, all_results):
    data = []
    for i in range(len(all_results)):
        data_by_method = []
        for j in range(len(all_results)):
            if i != j:
                # print(all_results[i])
                res = stats.wilcoxon(all_results[i], all_results[j])[1]
                data_by_method.append(res)
            else:
                data_by_method.append(np.inf)
        data.append(data_by_method)

    # print(data)
    # ax = sns.heatmap(data, linewidth=0.5, square=True)
    # plt.show()
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(data, xticklabels=labels, yticklabels=labels, mask=mask,
                         vmax=.3, square=True, cmap="YlGnBu", annot=True, fmt=".3f")
        plt.title(title)
        plt.show()


def plot_fpr_tpr_stacked_bars(title, data):
    data.plot(kind='bar', stacked=True, color=['darkred', 'darkblue'])

    plt.title(title)
    plt.xlabel('Methods')
    plt.ylabel('Percentage')
    plt.show()


def plot_box_plot_f1(title, data, labels):
    fig1, ax1 = plt.subplots(frameon=False)
    ax1.set_title(title)
    ax1.boxplot(data, labels=labels)
    plt.xlabel("Methods")
    plt.ylabel("F1")
    plt.show()


def plot_results(ood_type, ood_config):
    names = ['OTB', 'MSP', 'Max logit', 'Energy', 'SENA']  # 'Baseline',
    ood_title = ''
    # Loading novelty results
    if ood_type == 'novelty':
        ood_title = "Novelty class"
    elif ood_type == 'adversarial':
        ood_title = "Adversarial attack"
    if ood_type == 'distributional_shift':
        ood_title = "Distributional shift"

    ml_results = load_val('baseline', ood_type, ood_config)
    oob_results = load_val('oob', ood_type, ood_config)
    msp_results = load_val('msp', ood_type, ood_config)
    maxlog_results = load_val('maxlogit', ood_type, ood_config)
    energy_results = load_val('energy', ood_type, ood_config)
    sena_results = load_val('sena', ood_type, ood_config)

    # loading MCC results
    ml_mcc_results = np.round_(ml_results['mcc'], decimals=2)
    oob_mcc_results = np.round_(oob_results['mcc'], decimals=2)
    msp_mcc_results = np.round_(msp_results['mcc'], decimals=2)
    maxlog_mcc_results = np.round_(maxlog_results['mcc'], decimals=2)
    energy_mcc_results = np.round_(energy_results['mcc'], decimals=2)
    sena_mcc_results = np.round_(sena_results['mcc'], decimals=2)

    all_results_novelty = np.vstack([oob_mcc_results, msp_mcc_results,
                                     maxlog_mcc_results, energy_mcc_results, sena_mcc_results])
    ### Avg ranks
    arr_ranks = calculate_ranks(all_results_novelty)
    num_datasets = len(arr_ranks)
    avg_ranks = np.average(arr_ranks, axis=0)
    print("Num of datasets and average ranks:", num_datasets, np.round_(avg_ranks, decimals=1))

    ### MCC latex tables
    experiments = []
    for id_dataset, arr_ood_variant in ood_config.items():
        for ood_variant in arr_ood_variant:
            experiments.append('{} - {}'.format(id_dataset.upper(), ood_variant.upper()))
    df = pd.DataFrame(dict(Experiments=experiments,
                           Baseline=ml_mcc_results,
                           OOB=oob_mcc_results,
                           MSP=msp_mcc_results,
                           Maxlogit=maxlog_mcc_results,
                           Energy=energy_mcc_results,
                           SENA=sena_mcc_results,
                           # Averagerank=np.round_(avg_ranks, decimals=1)
                           ))
    latex_file = 'latex'
    df.to_latex(os.path.join(latex_file, '{}.tex'.format(ood_type)), index=False)

    ### Wilcoxon from MCC (plot)
    # heatmap of p-values calculated from Wilcoxon signed-rank tests goes here
    title = "P-values for {} experiments.".format(ood_title)
    plot_statistical_test_heatmap(title, names, all_results_novelty)

    ### F1 Boxplot
    f1_data = [# np.round_(ml_results['macro_f1'], decimals=2),
               np.round_(oob_results['macro_f1'], decimals=2),
               np.round_(msp_results['macro_f1'], decimals=2),
               np.round_(maxlog_results['macro_f1'], decimals=2),
               np.round_(energy_results['macro_f1'], decimals=2),
               np.round_(sena_results['macro_f1'], decimals=2)
               ]
    plot_box_plot_f1('F1 boxplots for {} experiments.'.format(ood_title), f1_data, names)

    ### Stacked barplots for FPR and FNR
    arr_avg_fpr = []
    arr_avg_fnr = []

    fpr_multiclass = np.round_(ml_results['fpr'], decimals=2)
    fnr_multiclass = np.round_(oob_results['fnr'], decimals=2)

    arr_fpr = [# np.average(fpr_multiclass),  # baseline
               np.round_(oob_results['fpr'], decimals=2),  # oob
               np.round_(msp_results['fpr'], decimals=2),  # msp
               np.round_(maxlog_results['fpr'], decimals=2),  # max logit
               np.round_(energy_results['fpr'], decimals=2),  # energy
               np.round_(sena_results['fpr'], decimals=2)  # sena
               ]
    arr_fnr = [# np.average(fnr_multiclass),  # baseline
               np.round_(oob_results['fnr'], decimals=2),  # oob
               np.round_(msp_results['fnr'], decimals=2),  # msp
               np.round_(maxlog_results['fnr'], decimals=2),  # max logit
               np.round_(energy_results['fnr'], decimals=2),  # energy
               np.round_(sena_results['fnr'], decimals=2)  # sena
               ]

    for fpr, fnr in zip(arr_fpr, arr_fnr):
        arr_avg_fpr.append(np.average(fpr))
        arr_avg_fnr.append(np.average(fnr))

    title = 'False positive/negative rates for {} experiments.'.format(ood_title)
    data = pd.DataFrame({'Average false positive rate': arr_avg_fpr,
                         'Average false negative rate': arr_avg_fnr},
                        index=names)

    plot_fpr_tpr_stacked_bars(title, data)


if __name__ == '__main__':
    arr_ood_exp_config = {
        "novelty": {"cifar10": ["svhn", "lsun", "tiny_imagenet", "gtsrb", "cifar100", "fractal"],
                    "svhn": ["cifar10", "lsun", "tiny_imagenet", "gtsrb", "cifar100", "fractal"],
                    "mnist": ["fashion_mnist", "emnist", "asl_mnist", "simpsons_mnist"]},
        "adversarial": {"cifar10": ["fgsm", "deepfool", "pgd"], "svhn": ["fgsm", "deepfool", "pgd"]
                        },
        "distributional_shift": {"cifar10": ["blur", "brightness", "pixelization", "shuffle_pixels",
                                             "contrast", "rotate", "saturation", "opacity"],
                                 "svhn": ["blur", "brightness", "pixelization", "shuffle_pixels",
                                          "contrast", "rotate", "saturation", "opacity"]
                                 }
    }
    # uncomment this line for writing the results in the csv folder
    for ood_type, ood_config in arr_ood_exp_config.items():
        # main(ood_type, ood_config)
        plot_results(ood_type, ood_config)
