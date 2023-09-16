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

    default_heatmap_kwargs = dict(
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar=False,)

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    # Plot the ground truth
    gt_data = sio.loadmat(args.gt_path)['truth']
    sns.heatmap(gt_data[::2, ::2],
        ax=ax1,
        **default_heatmap_kwargs)
    ax1.set_title('Ground truth')

    # Plot the BoW results
    bow_data = np.loadtxt(args.eval_path)
    # Take the lower triangle only
    bow_data = np.tril(bow_data, 0)
    sns.heatmap(bow_data,
        ax=ax2,
        vmax=0.2,
        **default_heatmap_kwargs)
    ax2.set_title('ORB + BoW')

    plt.tight_layout()
    plt.savefig(args.eval_path.replace('.txt', '.png'), bbox_inches='tight')
    plt.show()