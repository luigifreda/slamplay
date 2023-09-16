#!/usr/bin/env python3

import argparse
import os
import sys

import matplotlib

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.io as sio
import numpy as np

from paths import * 

# from https://github.com/nicolov/simple_slam_loop_closure/
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-path',
                        default=get_new_college_dataset_path() + '/NewCollegeGroundTruth.mat')
    parser.add_argument('--eval-path',
                        default=get_confusion_matrix_path())
    args = parser.parse_args()

    gt_data = sio.loadmat(args.gt_path)['truth'][::2, ::2]

    bow_data = np.loadtxt(args.eval_path)
    # Take the lower triangle only
    bow_data = np.tril(bow_data, -1)

    prec_recall_curve = []

    for thresh in np.arange(0, 0.09, 0.002):
        # precision: fraction of retrieved instances that are relevant
        # recall: fraction of relevant instances that are retrieved
        true_positives = (bow_data > thresh) & (gt_data == 1)
        all_positives = (bow_data > thresh)

        try:
            precision = float(np.sum(true_positives)) / np.sum(all_positives)
            recall = float(np.sum(true_positives)) / np.sum(gt_data == 1)

            prec_recall_curve.append([thresh, precision, recall])
        except:
            break

    prec_recall_curve = np.array(prec_recall_curve)

    plt.plot(prec_recall_curve[:, 1], prec_recall_curve[:, 2])

    for thresh, prec, rec in prec_recall_curve[5::5]:
        plt.annotate(
            str(thresh),
            xy=(prec, rec),
            xytext=(8, 8),
            textcoords='offset points')

    plt.xlabel('Precision', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.eval_path.replace('.txt', '_prec_recall.png'), bbox_inches='tight')
    plt.show()