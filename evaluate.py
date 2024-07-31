import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import matplotlib
import argparse

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

IN_TRAINING_COLOR = 'dodgerblue'
SUPPORT_COLOR = 'black'
OUT_TRAINING_COLOR = 'tomato'

def classify(method, in_samples, out_samples):
    y_true = [1] * len(in_samples) + [0] * len(out_samples)
    y_score = list(map(method, tqdm(in_samples, desc=f'in samples'))) + \
              list(map(method, tqdm(out_samples, desc=f'out samples')))

    return y_true, y_score


def plot_roc(y_true, y_score, loglog=False, name=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plot_fn = plt.loglog if loglog else plt.plot
    plot_fn(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color=IN_TRAINING_COLOR, alpha=0.7, linewidth=2.3)
    plot_fn([0, 1], [0, 1], linestyle=':', linewidth=0.9, color=SUPPORT_COLOR, dashes=(2, 10))
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.gca().set_aspect('equal')
    plt.xlabel('False Positive Rate' + (' (log scale)' if loglog else ''))
    plt.ylabel('True Positive Rate' + (' (log scale)' if loglog else ''))
    plt.title(('Log-Log ' if loglog else '') + f'Receiver Operating Characteristic' + ( f' for {name}' if name is not None else ''))
    plt.legend(loc='lower right')

def plot_hist(y_true, y_score):
    in_scores = np.array(y_score)[np.array(y_true, dtype=bool)]
    out_scores = np.array(y_score)[~np.array(y_true, dtype=bool)]

    x_min = np.min([in_scores, out_scores])
    x_max = np.max([in_scores, out_scores])

    bin_width = 0.2

    bins = np.arange(x_min, x_max+bin_width, bin_width)
    if len(bins) < 1:
        bins = 30
    print(bins)
    plt.hist(in_scores, bins=bins, alpha=0.75, label='In-Training set', color=IN_TRAINING_COLOR, histtype='step', linewidth=1.8)
    plt.hist(out_scores, bins=bins, alpha=0.75, label='Out-of-Training set', color=OUT_TRAINING_COLOR, histtype='step', linewidth=1.8)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend(loc='upper left')


def auc_for_method(y_true, y_score, name):
    results = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
    results.to_csv(f'results_{name}.csv', index=False)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    tpr_at_1_fpr = np.interp(0.01, fpr, tpr)
    tpr_at_5_fpr = np.interp(0.05, fpr, tpr)

    with open(f'metrics_{name}.csv', 'w') as f:
        f.write(f'AUC,{roc_auc}\n')
        f.write(f'TPR@1%FPR,{tpr_at_1_fpr}\n')
        f.write(f'TPR@5%FPR,{tpr_at_5_fpr}\n')

    return roc_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    from methods import alternative_1, alternative_2, mink_pp, mink_pp_filter
    METHODS = [("alt1", alternative_1), ("alt2", alternative_2), ("mink_pp", mink_pp), ("mink_pp_filter", mink_pp_filter)]

    training_logits = torch.load("MaybeBrokenEleutherAIpythia28bswj0419BookMIAtrain.pt", map_location=torch.device('cpu'))
    in_training_logits = training_logits["in"]
    out_of_training_logits = training_logits["out"]

    for (name, method) in METHODS:
        plt.figure()
        y_true, y_score = classify(method, in_training_logits, out_of_training_logits)
        roc_auc = auc_for_method(y_true, y_score, name)
        plt.title(f"AUC: {roc_auc:.4f}")
        # plot_hist(y_true, y_score)
        plot_roc(y_true, y_score, args.log)
        plt.savefig(f"plots/{name}.png")
        plt.clf()