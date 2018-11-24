r""" Prune weights from checkpoint file.

When training a model with, your training directory will
containing a GraphDef file (usually ending with the .pb or .pbtxt extension)
and a set of checkpoint files. We load both here and output a pruned
version of the GraphDef and checkpoint file.

As described in https://arxiv.org/abs/1608.08710.
  Pruning Filters for Efficient ConvNets
  Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, Hans Peter Graf

Usage:

    python compress.py \
        --input_graph /tmp/models/prediction_graph.pbtxt \
        --input_checkpoint /tmp/models/model.ckpt-XYZ \
        --output_dir /tmp/pruned_model \
        --config_path configs/compression/icnet_resnet_v1_prune_all.config \
        --skippable_nodes "Predictions/postrain/biases"
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

flags.DEFINE_string('input_checkpoint', None,
                    'TensorFlow variables file to load.')
flags.mark_flag_as_required('input_checkpoint')

flags.DEFINE_string('prune_config', None,
                    'The compression config to use to compression.')
flags.mark_flag_as_required('prune_config')

flags.DEFINE_float('compression_factor', 0.5,
                   'The compression factor to apply when prunin filters.')

flags.DEFINE_boolean('input_binary', False,
                     'Whether the input files are in binary format.')

flags.DEFINE_string('output_dir', '',
                    'Location to save prunned output checkpoints')

flags.DEFINE_string('skippable_nodes', '',
                    'Nodes to not validate when pruning.')

flags.DEFINE_boolean('interactive', False,
                     'Whether the input files are in binary format.')

flags.DEFINE_boolean('soft_apply', False,
                     'Simulate compression by setting weights to zero but '
                     'keeping the original shape of each variable.')


def main(unused_args):
    if (not tf.train.checkpoint_exists(FLAGS.input_checkpoint)
        or not tf.gfile.Exists(FLAGS.input_checkpoint + '.meta')):
        print('The input checkpoint prefix specified from '
              '`FLAGS.input_checkpoint` must point to a location with '
              'valid meta and data cehckpoint files.')
        return -1

    output_path_name = "pruned_model.ckpt"
    compression_config = compressor_pb2.CommpressionConfig()
    with tf.gfile.GFile(FLAGS.prune_config, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, compression_config)
    compression_strategy_config = compression_config.compression_strategy

    skippable_nodes = FLAGS.skippable_nodes.replace(" ", "").split(",")
    compression_fn = functools.partial(
        compressor_builder.build,
        compression_factor=FLAGS.compression_factor,
        skippable_nodes=skippable_nodes,
        compression_config=compression_strategy_config,
        interactive_mode=FLAGS.interactive,
        soft_apply=FLAGS.soft_apply)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    compressor = compression_fn()
    compressor.compress(FLAGS.input_checkpoint)
    compressor.save(
        output_checkpoint_dir=FLAGS.output_dir,
        output_checkpoint_name=output_path_name)


if __name__ == '__main__':
    tf.app.run()
