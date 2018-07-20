r""" Prune weights from checkpoint file.

When training a model with, your training directory will
containing a GraphDef file (usually ending with the .pb or .pbtxt extension)
and a set of checkpoint files. We load both here and output a pruned
version of the GraphDeg and checkpoint file.

Based on the paper:

Usage:

prune_weights.py \
--input_graph=/tmp/model/my_graph.pb \
--input_checkpoint=/tmp/model/model.ckpt-1000 \
--output_graph=/tmp/frozen_graph.pb \
--output_node_names=output_node \

TEMP Example:

    python compress.py \
        --input_graph=tmp/weights/prediction_graph.pbtxt \
        --input_checkpoint=tmp/weights/model.ckpt-57021 \
        --output_dir=tmp/weights_pruned \
        --config_pat=configs/icnet_resnet_v1_pruner.config \
        --skippable_nodes="Predictions/postrain/biases,CascadeFeatureFusion_0/AuxOutput/biases, CascadeFeatureFusion_1/AuxOutput/biases"

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import functools
from google.protobuf import text_format
import tensorflow as tf

from protos import compressor_pb2
from builders import compressor_builder

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_float('compression_factor', 0.5,
                   'The compression factor to apply when prunin filters.')

flags.DEFINE_string('input_graph', '',
                    'TensorFlow \'GraphDef\' file to load.')

flags.DEFINE_string('input_checkpoint', '',
                    'TensorFlow variables file to load.')

flags.DEFINE_boolean('input_binary', True,
                     'Whether the input files are in binary format.')

flags.DEFINE_string('output_dir', '',
                    'Location to save prunned output checkpoints')

flags.DEFINE_string('config_path', '',
                    'The compression config to use to compression.')

flags.DEFINE_string('skippable_nodes', '',
                    'Nodes to not validate when pruning.')

flags.DEFINE_boolean('interactive', False,
                     'Whether the input files are in binary format.')


def main(unused_args):
    if not tf.gfile.Exists(FLAGS.input_graph):
        print('The `input_graph` specified does not exist.')
        return -1

    output_path_name = "prunned_model.ckpt"
    compression_config = compressor_pb2.CommpressionConfig()
    with tf.gfile.GFile(FLAGS.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, compression_config)
    compression_strategy_config = compression_config.compression_strategy

    skippable_nodes = FLAGS.skippable_nodes.replace(" ", "").split(",")
    compression_fn = functools.partial(
        compressor_builder.build,
        compression_factor=FLAGS.compression_factor,
        skippable_nodes=skippable_nodes,
        compression_config=compression_strategy_config,
        interactive_mode=FLAGS.interactive)

    input_graph_def = tf.GraphDef()
    mode = "rb" if FLAGS.input_binary else "r"
    with tf.gfile.FastGFile(FLAGS.input_graph, mode) as f:
        if FLAGS.input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    compressor = compression_fn()
    compressor.compress(input_graph_def, FLAGS.input_checkpoint)
    compressor.save(
        output_checkpoint_dir=FLAGS.output_dir,
        output_checkpoint_name=output_path_name)

if __name__ == '__main__':
    tf.app.run()
