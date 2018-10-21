r"""Export segmentation model to a serialized frozen graph file."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from google.protobuf import text_format

from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos

from protos import pipeline_pb2
from builders import model_builder
from libs.constants import CITYSCAPES_LABEL_COLORS
from libs.exporter import deploy_segmentation_inference_graph


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_shape', None,
                    'The shape to use for the placeholder tensor. This should '
                    'be in the form of [batch, height, width, channels] or '
                    '[height, width, channels].')

flags.DEFINE_string('pad_to_shape', None,
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs.')

flags.DEFINE_boolean('output_colours', False,
                     'Whether the output should be RGB image.')


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with tf.Session() as sess:
            saver = tf.train.Saver(saver_def=input_saver_def,
                              save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)


def profile_inference_graph(graph):
    tfprof_vars_option = (
        tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    tfprof_flops_option = tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS

    tfprof_vars_option['trim_name_regexes'] = ['.*BatchNorm.*']
    tfprof_flops_option['trim_name_regexes'] = [
        '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
    ]

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        graph,
        tfprof_options=tfprof_vars_option)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        graph,
        tfprof_options=tfprof_flops_option)


def export_inference_graph(pipeline_config,
                           trained_checkpoint_prefix,
                           output_directory,
                           input_shape=None,
                           pad_to_shape=None,
                           output_colours=False,
                           output_collection_name='predictions'):

    _, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                   'frozen_inference_graph.pb')
    eval_graphdef_path = os.path.join(output_directory,
                                    'export_graph.pbtxt')
    saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=segmentation_model,
        input_shape=input_shape,
        pad_to_shape=pad_to_shape,
        label_color_map=(CITYSCAPES_LABEL_COLORS
            if output_colours else None),
        output_collection_name=output_collection_name)

    profile_inference_graph(tf.get_default_graph())

    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()

    graph_def = tf.get_default_graph().as_graph_def()
    f = tf.gfile.FastGFile(eval_graphdef_path, "w")
    f.write(str(graph_def))

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=trained_checkpoint_prefix)

    output_node_names = outputs.name.split(":")[0]

    freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=trained_checkpoint_prefix,
        output_graph=frozen_graph_path,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes='')

    print("Done!")


def main(_):
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')]
    else:
        input_shape = None

    pad_to_shape = None
    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.pad_to_shape.split(',')]

    export_inference_graph(pipeline_config,
                           FLAGS.trained_checkpoint,
                           FLAGS.output_dir, input_shape,
                           pad_to_shape,FLAGS.output_colours)


if __name__ == '__main__':
    tf.app.run()
