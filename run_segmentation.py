r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
from scipy import misc
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph


LABEL_COLORS = [
    (128,  64, 128),        # road
    (244,  35, 231),        # sidewalk
    ( 69,  69,  69),        # building
    (102, 102, 156),        # wall
    (190, 153, 153),        # fence
    (153, 153, 153),        # pole
    (250, 170,  29),        # traffic light
    (219, 219,   0),        # traffic sign
    (106, 142,   3),        # vegetation
    (152, 250, 152),        # terrain
    ( 69, 129, 180),        # sky
    (219,  19,  60),        # person
    (255,   0,   0),        # rider
    (  0,   0, 142),        # car
    (  0,   0,  69),        # truck
    (  0,  60, 100),        # bus
    (  0,  79, 100),        # train
    (  0,   0, 230),        # motocycle
    (119,  10,   3)]        # bicycle

VALID_IMAGE_FILE_EXT = ['.JPG', '.JPEG', '.PNG']


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None,
                    'Path to an image or a directory of images.')

flags.DEFINE_string('input_shape', None,
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

flags.DEFINE_string('pad_to_shape', None,
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')


def _valid_file_ext(input_path, valid_ext=VALID_IMAGE_FILE_EXT):
    ext = os.path.splitext(input_path)[-1].upper()
    return ext in valid_ext


def _get_images_from_path(input_path):
    image_file_paths = []
    if os.path.isdir(input_path):
        for dirpath,_,filenames in os.walk(input_path):
            for f in filenames:
                file_path = os.path.abspath(os.path.join(dirpath, f))
                if not _valid_file_ext(file_path):
                    raise ValueError('File must be JPG or PNG.')
                image_file_paths.append(file_path)
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    print("found {} images...".format(len(image_file_paths)))
    return image_file_paths


def run_inference_graph(pipeline_config, trained_checkpoint_prefix,
                           output_directory, input_images, input_shape,
                           pad_to_shape):

    num_classes, segmentation_model = model_builder.build(pipeline_config.model,
                                             is_training=False)
    tf.gfile.MakeDirs(output_directory)
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=segmentation_model,
        input_shape=input_shape,
        label_color_map=LABEL_COLORS,
        pad_to_shape=pad_to_shape,
        num_classes=num_classes,
        output_collection_name="inference_op")

    image_mode = "RGB" if input_shape[-1] == 3 else "L"
    with tf.Session() as sess:
        input_graph_def = tf.get_default_graph().as_graph_def()
        saver = tf.train.Saver()
        saver.restore(sess, trained_checkpoint_prefix)

        for image_path in input_images:
            image_raw = misc.imread(image_path, mode=image_mode)

            start_time = timeit.default_timer()
            predications = sess.run(outputs,
                feed_dict={placeholder_tensor: image_raw})
            elapsed = timeit.default_timer() - start_time

            print('wall time: {}'.format(elapsed))
            filename = os.path.basename(image_path)
            save_location = os.path.join(output_directory, filename)
            misc.imsave(save_location, predications[0])


def main(_):
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.input_shape.split(',')]
    else:
        raise ValueError('Must supply `input_shape`')

    pad_to_shape = None
    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
            for dim in FLAGS.pad_to_shape.split(',')]

    if (not os.path.isdir(FLAGS.input_path) and
            not os.path.isfile(FLAGS.input_path)):
        raise ValueError("`input_path` must be a valid directory or image file")

    input_images = _get_images_from_path(FLAGS.input_path)
    run_inference_graph(pipeline_config,
                        FLAGS.trained_checkpoint,
                        FLAGS.output_dir, input_images, input_shape,
                        pad_to_shape)


if __name__ == '__main__':
    tf.app.run()
