import os
import collections

import pickle
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_float('compression_factor_step', 0.1,
                   'The compression factor to apply when prunin filters.')

flags.DEFINE_string('compression_state', '', 'Pickle file from other script')



def plot_compression_sensitivity(results_dict):
    # Get flags
    compression_factor_step = FLAGS.compression_factor_step
    compression_factor_list = np.arange(0.1, 1.1, compression_factor_step)[::-1]
    # Setup Plot styles
    fig = plt.figure(
        figsize=(20,10)
    )
    plt.style.use('seaborn-bright')
    # Build plot
    for layer_name, acc_dict in results_dict.items():
        eval_values = [acc_dict[str(cf)] for cf in compression_factor_list]
        plt.plot(compression_factor_list, eval_values,
            label=layer_name,
            marker="o",
            markeredgewidth=0.5,
            markeredgecolor=(0,0,0,1)
            # color=colors[layer_name]
            )
    plt.ylabel('Accuracy (mIoU)')
    plt.xlabel('Filter Sparsity (%)')
    plt.title('Pruning Sensitivity, ICNet Cityscapes')
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5)
    # plt.legend(loc='lower center', ncol=2, mode="expand", borderaxespad=0.)
    ax.invert_xaxis()
    ax.autoscale(enable=True, axis='x', tight=True)
    plt.show()


def main():
    if FLAGS.compression_state == '':
        raise ValueError('Please provide the pickle state file...')

    metrics_dict = {}
    with open(FLAGS.compression_state, 'rb') as f:
        metrics_dict = pickle.load(f)
    print(metrics_dict)
    plot_compression_sensitivity(metrics_dict)


if __name__ == '__main__':
    main()
