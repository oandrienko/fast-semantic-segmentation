r"""
python run_compress_sensitivity.py \
    --input_graph tmp/weights/prediction_graph.pb \
    --input_checkpoint tmp/weights/model.ckpt-57021 \
    --output_dir tmp/sensitivity \
    --prune_config_path configs/pruner/icnet_resnet_v1_pruner_v1.config \
    --skippable_nodes "Predictions/postrain/biases" \
    --eval_config_path configs/icnet_1.0_1025_resnet_v1.config
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import functools
import numpy as np
from google.protobuf import text_format
import tensorflow as tf

from protos import pipeline_pb2
from protos import compressor_pb2
from builders import model_builder
from builders import dataset_builder
from builders import compressor_builder

from libs.evaluator import eval_segmentation_model_once


tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_float('compression_factor_step', 0.1,
                   'The compression factor to apply when prunin filters.')

flags.DEFINE_string('input_graph', '',
                    'TensorFlow \'GraphDef\' file to load.')

flags.DEFINE_string('input_checkpoint', '',
                    'TensorFlow variables file to load.')

flags.DEFINE_boolean('input_binary', True,
                     'Whether the input files are in binary format.')

flags.DEFINE_string('output_dir', '',
                    'Location to save prunned output checkpoints')

flags.DEFINE_string('eval_config_path', '',
                    'The compression config to use to compression.')

flags.DEFINE_string('prune_config_path', '',
                    'The compression config to use to compression.')

flags.DEFINE_string('skippable_nodes', '',
                    'Nodes to not validate when pruning.')

flags.DEFINE_boolean('keep_all_checkpoints', False,
                     'Whether to keep checkpoints and not delete them.')


def log_dir_name(layer_name, percent_kept, logdir):
    layer_name = layer_name.replace('/','_')
    trial_path = "scale_{}".format(percent_kept)
    log_dir = os.path.join(logdir, layer_name, trial_path)
    return log_dir


def create_compressor(compression_config,
                     compression_factor,
                     skippable_nodes,
                     soft_apply):
    skippable_nodes = skippable_nodes.replace(" ", "").split(",")
    compression_fn = functools.partial(
        compressor_builder.build,
        compression_factor=compression_factor,
        skippable_nodes=skippable_nodes,
        compression_config=compression_config,
        interactive_mode=False,
        soft_apply=True)

    input_graph_def = tf.GraphDef()
    mode = "rb" if FLAGS.input_binary else "r"
    with tf.gfile.FastGFile(FLAGS.input_graph, mode) as f:
        if FLAGS.input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)

    compressor = compression_fn()
    compressor.compress(
        input_graph_def, FLAGS.input_checkpoint, skip_apply=True)
    return compressor


def run_compression(pruner_spec,
                    compression_config,
                    skippable_nodes,
                    compression_factor):
    layer_name = pruner_spec.target

    compressor = create_compressor(
        compression_config.compression_strategy,
        compression_factor,
        skippable_nodes,
        soft_apply=True)
    compressor.pruner_specs = [pruner_spec]
    compressor._apply_pruner_specs(
        compressor.pruner_specs)

    output_dir_name = log_dir_name(
        layer_name, compression_factor, FLAGS.output_dir)
    tf.gfile.MakeDirs(output_dir_name)
    output_checkpoint_name = "pruned_model.ckpt"
    output_path_name = os.path.join(output_dir_name, output_checkpoint_name)
    compressor.save(
        output_checkpoint_dir=output_dir_name,
        output_checkpoint_name=output_checkpoint_name)
    return output_path_name, output_dir_name


def run_eval(curr_checkpoint,
             eval_dir,
             model_config,
             input_config,
             eval_config):
    create_input_fn = functools.partial(
        dataset_builder.build,
        input_reader_config=input_config)
    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False)

    eval_input_type = eval_config.eval_input_type
    input_type = eval_input_type.WhichOneof('eval_input_type_oneof')
    if input_type == 'cropped_eval_input':
        cropped_eval_input = eval_input_type.cropped_eval_input
        input_dims = (cropped_eval_input.height,
                      cropped_eval_input.width)
        cropped_evaluation = True
    elif input_type == 'padded_eval_input':
        padded_eval_input = eval_input_type.padded_eval_input
        input_dims = (padded_eval_input.height,
                      padded_eval_input.width)
        cropped_evaluation = False
    else:
        raise ValueError('Must specify an `eval_input_type` for evaluation.')

    metrics = eval_segmentation_model_once(
        curr_checkpoint,
        create_model_fn,
        create_input_fn,
        input_dims,
        eval_config=eval_config,
        eval_dir=eval_dir,
        cropped_evaluation=False,
        image_summaries=False,
        verbose=True)

    return metrics

def main(unused_args):
    if not tf.gfile.Exists(FLAGS.input_graph):
        print('The `input_graph` specified does not exist.')
        return -1
    output_path_name = "prunned_model.ckpt"

    # Compression_config
    compression_config = compressor_pb2.CommpressionConfig()
    with tf.gfile.GFile(FLAGS.prune_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, compression_config)
    compression_strategy_config = compression_config.compression_strategy

    input_graph_def = tf.GraphDef()
    mode = "rb" if FLAGS.input_binary else "r"
    with tf.gfile.FastGFile(FLAGS.input_graph, mode) as f:
        if FLAGS.input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)

    # Eval_config
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.eval_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    eval_config = pipeline_config.eval_config
    input_config = pipeline_config.eval_input_reader
    model_config = pipeline_config.model

    # Setup experiment stuff
    compression_factor_step = FLAGS.compression_factor_step
    if compression_factor_step > 1.0 or compression_factor_step < 0:
        raise ValueError('`compression_factor_step` must be between [0,1].')
    compression_factor_list = np.arange(0.1, 1.1, compression_factor_step)[::-1]

    # Get pruner specs
    init_compressor = create_compressor(
        compression_config=compression_strategy_config,
        compression_factor=1.0,
        skippable_nodes=FLAGS.skippable_nodes,
        soft_apply=True)
    init_compressor.compress(
        input_graph_def, FLAGS.input_checkpoint, skip_apply=True)
    all_pruner_specs = init_compressor.pruner_specs

    # Run experiment
    save_file_path = os.path.join(FLAGS.output_dir, "compress_state.pkl")
    METRICS_RESULTS = {}
    for pruner_spec in all_pruner_specs:
        print("")
        print("")
        print("STARTING NEW LAYER EXPERIMENT: ", pruner_spec.target)
        layer_results = {}
        for compression_factor in compression_factor_list:
            tf.reset_default_graph()
            print(" - pruner compression_factor: ", compression_factor)
            curr_output_ckpt_name, curr_output_dir_name = run_compression(
                pruner_spec=pruner_spec,
                compression_config=compression_config,
                compression_factor=compression_factor,
                skippable_nodes=FLAGS.skippable_nodes)
            tf.reset_default_graph()
            output_metric = run_eval(
                curr_output_ckpt_name,
                model_config=model_config,
                input_config=input_config,
                eval_config=eval_config,
                eval_dir=curr_output_dir_name)
            layer_results[str(compression_factor)] = output_metric
            print(" - eval result: ", output_metric)
        METRICS_RESULTS[pruner_spec.target] = layer_results
        with open(save_file_path, 'wb') as f:
            pickle.dump(METRICS_RESULTS, f, pickle.HIGHEST_PROTOCOL)
        if not FLAGS.keep_all_checkpoints:
            tf.gfile.DeleteRecursively(curr_output_dir_name)
    print("")
    print("DONE!")

if __name__ == '__main__':
    tf.app.run()
