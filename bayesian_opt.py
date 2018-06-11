r"""Search for Hyperparameters using Baysian Optimization

Example Usage:

    python bayesian_opt.py
        --config_path=configs/bayes_icnet_resnet_v1_cityscapes.config \
        --logdir=tmp/test_baye \
        --num_trials=11 | tee baye_test_log.log
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import time
import functools
import math
import pprint
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.protobuf import text_format

from skopt import gp_minimize, load as skopt_load_state
from skopt.plots import plot_convergence
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import callbacks
from skopt.callbacks import CheckpointSaver

from protos import pipeline_pb2
from core.evaluator import eval_segmentation_model_once
from core.trainer import train_segmentation_model

tf.logging.set_verbosity(tf.logging.INFO)
# tf.logging.set_verbosity(tf.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags

FLAGS = flags.FLAGS

# All Hyperparameter definitions

LR_RANGE                = (1e-5, 1e-2)
MAIN_LOSS_WT            = (0.5,  1.5)
FIRST_AUX_LOSS_WT       = (0.1,  0.8)
SECOND_AUX_LOSS_WT      = (0.1,  0.8)
L2_REGULARIZATION       = (1e-5, 1e-3)

DEFAULTS = [1e-3, 1.0, 0.4, 0.15, 1e-4]

SEED = 777

# Optimization settings

flags.DEFINE_integer('num_trials', 15,
                     'Trail to run for the Baysian Optimization process.')

flags.DEFINE_string('from_skopt_checkpoint', '',
                   'Initialize hyperparameter optimizer from Skopt checkpoint.')

# Distributed training settings

flags.DEFINE_integer('num_clones', 1,
                     'Number of model clones to deploy to each worker replica.'
                     'This should be greater than one if you want to use '
                     'multiple GPUs located on a single machine.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1,
                     'Number of worker replicas. This typically corresponds '
                     'to the number of machines you are training on. Note '
                     'that the training will be done asynchronously.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker. It is '
                     'reccomended to use num_ps_tasks=num_replicas/2.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID. Should increment per worker '
                     'replica added to achieve between graph replication.')

# Training configuration settings

flags.DEFINE_string('config_path', '',
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('config_path')

flags.DEFINE_string('logdir', '',
                    'Directory to save the checkpoints and training summaries.')
flags.mark_flag_as_required('logdir')

flags.DEFINE_integer('save_every_n_hours', 999,
                     'Time between successive saves of a checkpoint')

# GLOBALS

BEST_ACCURACY = 0.0
TRIAL_RUN = 0


def log_dir_name(lr, mbl_wt, fbl_wt, sbl_wt, wr, logdir, trial):
    trial_path = "T{0}_lr_{1:.0e}_mblwt_{2:.0e}_fblwt_{3:.0e}_sblwt_{4:.0e}_wr_{5:.0e}".format(
        trial, lr, mbl_wt, fbl_wt, sbl_wt, wr)
    log_dir = os.path.join(logdir, trial_path)
    return log_dir


def fitness(hparams,
            root_logdir,
            train_config,
            train_input_config,
            eval_config,
            eval_input_config,
            model_config,
            master,
            task,
            is_chief,
            startup_delay_steps,
            num_clones,
            num_replicas,
            clone_on_cpu,
            num_ps_tasks,
            save_every_n_hours):

    global BEST_ACCURACY
    global TRIAL_RUN

    lr, mbl_wt, fbl_wt, sbl_wt, wr = hparams

    # FIX @oandrien
    # SUPER HACKY - requires fixed config...
    (train_config.optimizer.momentum_optimizer.learning_rate \
        .polynomial_decay_learning_rate.initial_learning_rate) = lr
    (model_config.icnet.loss.main_loss_weight) = mbl_wt
    (model_config.icnet.loss.second_branch_loss_weight) = fbl_wt
    (model_config.icnet.loss.first_branch_loss_weight) = sbl_wt
    (model_config.icnet.hyperparams.regularizer.l2_regularizer.weight) = wr

    print()
    print('Starting New Trial...')
    print()
    print('Trail {}'.format(TRIAL_RUN))
    print('Learning Rate: {0:.1e}'.format(lr))
    print('Main Branch Loss Weight:', mbl_wt)
    print('First Branch Aux Loss Weight:', fbl_wt)
    print('Second Branch Aux Loss Weight:', sbl_wt)
    print('L2 Weight Regularization Value:', wr)
    print()

    print(train_config)
    time.sleep(3) #
    print(model_config)
    time.sleep(3) #

    train_dir = log_dir_name(lr, mbl_wt, fbl_wt, sbl_wt, wr,
                            logdir=root_logdir, trial=TRIAL_RUN)
    eval_dir = train_dir+'_EVAL'
    tf.gfile.MakeDirs(train_dir)
    tf.gfile.MakeDirs(eval_dir)

    total_loss = train_segmentation_model(
                    train_config,
                    train_input_config,
                    model_config,
                    master=master,
                    task=task,
                    is_chief=is_chief,
                    startup_delay_steps=startup_delay_steps,
                    train_dir=train_dir,
                    num_clones=num_clones,
                    num_worker_replicas=num_replicas,
                    clone_on_cpu=clone_on_cpu,
                    replica_id=task,
                    num_replicas=num_replicas,
                    num_ps_tasks=num_ps_tasks,
                    save_every_n_hours=save_every_n_hours)
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
        raise ValueError('`latest_checkpoint` does not exist... fuck')

    print()
    print("Running Eval")
    print("Using checkpoint {}".format(latest_checkpoint))
    print()
    time.sleep(5)

    acc_summary = eval_segmentation_model_once(
                    eval_config,
                    eval_input_config,
                    model_config,
                    checkpoint_path=latest_checkpoint,
                    train_dir=train_dir,
                    eval_dir=eval_dir,
                    master=master)
    accuracy = acc_summary[0]

    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print("Done trial!")
    print()
    time.sleep(5)

    # Clear variables, grpah, ops for next run
    tf.reset_default_graph()

    # Update globals and return results for trial
    TRIAL_RUN += 1
    if accuracy > BEST_ACCURACY:
        BEST_ACCURACY = accuracy
    return -accuracy


def optimize_over_all(fitness_fn,
                      dimensions_fn,
                      trials=15,
                      init_x0=None,
                      init_y0=None,
                      logdir=''):

    dimensions = dimensions_fn()

    gp_ckpt_path = os.path.join(logdir, "skopt_checkpoint.pkl")
    checkpoint_saver = CheckpointSaver(gp_ckpt_path, compress=9)

    search_result = gp_minimize(func=fitness_fn,
                                 dimensions=dimensions,
                                 acq_func='EI',
                                 n_calls=trials,
                                 x0=init_x0, # default_params
                                 callback=[checkpoint_saver],
                                 random_state=SEED, # need this
                                 verbose=True)
    plot_convergence(search_result)

    # Print best results
    print()
    print("**BEST HParams RESULTS**")
    print(search_result.x)
    print('Learning Rate: {0:.1e}'.format(search_result.x[0]))
    print('Main Branch Loss Weight: {0:.1e}'.format(search_result.x[1]))
    print('First Branch Aux Loss Weight: {0:.1e}'.format(search_result.x[2]))
    print('Second Branch Aux Loss Weight: {0:.1e}'.format(search_result.x[3]))
    print('L2 Weight Regularization Value: {0:.1e}'.format(search_result.x[4]))

    # Save result of the results here:
    class_results_out = os.path.join(logdir, "_BAYES_RESULTS.class.pkl")
    with open(class_results_out, 'wb') as f:
        pickle.dump(search_result, f)
    val_results_out = os.path.join(logdir, "_BAYES_RESULTS.list.pkl")
    result_list = sorted(zip(search_result.func_vals, search_result.x_iters))
    with open(val_results_out, 'wb') as f:
        pickle.dump(result_list, f)

    print()
    print("**ALL RUNS**")
    pprint.pprint(result_list)


def build_skopt_hparams_fn():
    search_space = [

        Real(low=LR_RANGE[0], high=LR_RANGE[1],
             name='InitialLearninRate'),

        Real(low=MAIN_LOSS_WT[0], high=MAIN_LOSS_WT[1],
             name='MainBranchLossWeight'),

        Real(low=FIRST_AUX_LOSS_WT[0], high=FIRST_AUX_LOSS_WT[1],
             name='FirstBranchAuxLossWeight'),

        Real(low=SECOND_AUX_LOSS_WT[0], high=SECOND_AUX_LOSS_WT[1],
             name='SecondBranchAuxLossWeight'),

        Real(low=L2_REGULARIZATION[0], high=L2_REGULARIZATION[1],
            name='L2WeightRegulatization')

    ]
    return search_space


def main(_):
    tf.gfile.MakeDirs(FLAGS.logdir)
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    train_input_config = pipeline_config.train_input_reader
    eval_config = pipeline_config.eval_config
    eval_input_config = pipeline_config.eval_input_reader

    is_chief = (FLAGS.task == 0)

    fitness_fn = functools.partial(fitness,
            root_logdir=FLAGS.logdir,
            train_config=train_config,
            train_input_config=train_input_config,
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            master=FLAGS.master,
            task=FLAGS.task,
            is_chief=is_chief,
            startup_delay_steps=FLAGS.startup_delay_steps,
            num_clones=FLAGS.num_clones,
            num_replicas=FLAGS.num_replicas,
            clone_on_cpu=FLAGS.clone_on_cpu,
            num_ps_tasks=FLAGS.num_ps_tasks,
            save_every_n_hours=FLAGS.save_every_n_hours)

    x0 = DEFAULTS
    y0 = None
    if FLAGS.from_skopt_checkpoint:
        gp_states = skopt_load_state(FLAGS.from_skopt_checkpoint)
        x0 = gp_states.x_iters
        y0 = gp_states.func_vals

    optimize_over_all(
        fitness_fn,
        build_skopt_hparams_fn,
        trials=FLAGS.num_trials,
        init_x0=x0,
        init_y0=y0,
        logdir=FLAGS.logdir)


if __name__ == '__main__':
    tf.app.run()