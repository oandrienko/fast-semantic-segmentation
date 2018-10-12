r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
from PIL import Image
import tensorflow as tf
from google.protobuf import text_format

from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None,
                    'Path to an image or a directory of images.')

flags.DEFINE_string('input_shape', '1024,2048,3', # default Cityscapes values
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

flags.DEFINE_string('pad_to_shape', '1025,2049', # default Cityscapes values
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')


def _valid_file_ext(input_path):
    ext = os.path.splitext(input_path)[-1].upper()
    return ext in ['.JPG', '.JPEG', '.PNG']


def _get_images_from_path(input_path):
    image_file_paths = []
    if os.path.isdir(input_path):
        for dirpath,_,filenames in os.walk(input_path):
            for f in filenames:
                file_path = os.path.abspath(os.path.join(dirpath, f))
                if _valid_file_ext(file_path):
                    image_file_paths.append(file_path)
        if len(image_file_paths) == 0:
            raise ValueError('No images in directory. '
                             'Files must be JPG or PNG')
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    return image_file_paths


def run_inference_graph(model, trained_checkpoint_prefix,
                        input_images, input_shape, pad_to_shape,
                        label_color_map, output_directory):

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_shape,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)

    with tf.Session() as sess:
        input_graph_def = tf.get_default_graph().as_graph_def()
        saver = tf.train.Saver()
        saver.restore(sess, trained_checkpoint_prefix)

        for idx, image_path in enumerate(input_images):
            image_raw = np.array(Image.open(image_path))

            start_time = timeit.default_timer()
            predictions = sess.run(outputs,
                feed_dict={placeholder_tensor: image_raw})
            elapsed = timeit.default_timer() - start_time

            print('{}) wall time: {}'.format(elapsed, idx+1))
            filename = os.path.basename(image_path)
            save_location = os.path.join(output_directory, filename)

            predictions = predictions.astype(np.uint8)
            if len(label_color_map[0]) == 1:
               predictions = np.squeeze(predictions,-1)
            im = Image.fromarray(predictions[0])
            im.save(save_location, "PNG")


def main(_):
    output_directory = FLAGS.output_dir
    tf.gfile.MakeDirs(output_directory)
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    pad_to_shape = None
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.input_shape.split(',')]
    else:
        raise ValueError('Must supply `input_shape`')

    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.pad_to_shape.split(',')]

    input_images = _get_images_from_path(FLAGS.input_path)
    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_images, input_shape, pad_to_shape,
                        label_map, output_directory)

if __name__ == '__main__':
    tf.app.run()
