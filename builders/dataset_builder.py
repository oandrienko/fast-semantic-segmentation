r"""Builder for semantic segmentation dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from libs import standard_fields as fields
from protos import input_reader_pb2


slim = tf.contrib.slim
tfexample_decoder = slim.tfexample_decoder


def _create_tf_example_decoder():

    keys_to_features = {
        fields.TFRecordFields.image_encoded:
            tf.FixedLenFeature((), tf.string, ''),
        fields.TFRecordFields.image_format:
            tf.FixedLenFeature((), tf.string, ''),
        fields.TFRecordFields.image_filename:
            tf.FixedLenFeature((), tf.string, ''),
        fields.TFRecordFields.image_height:
            tf.FixedLenFeature((), tf.int64, 0),
        fields.TFRecordFields.image_width:
            tf.FixedLenFeature((), tf.int64, 0),
        fields.TFRecordFields.segmentation_class_encoded:
            tf.FixedLenFeature((), tf.string, ''),
        fields.TFRecordFields.segmentation_class_format:
            tf.FixedLenFeature((), tf.string, ''),
    }

    # Main GT Input and output tensors for full image segmentation task
    input_image = tfexample_decoder.Image(
        image_key=fields.TFRecordFields.image_encoded,
        format_key=fields.TFRecordFields.image_format,
        # shape=(1024, 2048, 3),  # TODO: Move this, it's CITYSCAPES SPECIFIC
        channels=3)
    output_mask = tfexample_decoder.Image(
        image_key=fields.TFRecordFields.segmentation_class_encoded,
        format_key=fields.TFRecordFields.segmentation_class_format,
        # shape=(1024, 2048, 1),  # TODO: Move this, it's CITYSCAPES SPECIFIC
        channels=1)

    items_to_handlers = {
        fields.GroundtruthFields.input_image_path:
            tfexample_decoder.Tensor('image/filename'),
        fields.GroundtruthFields.input_image_height:
            tfexample_decoder.Tensor('image/height'),
        fields.GroundtruthFields.input_image_width:
            tfexample_decoder.Tensor('image/width'),
        # Masks
        fields.GroundtruthFields.input_image: input_image,
        fields.GroundtruthFields.output_mask: output_mask
    }

    return tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


def _process_fn(tf_serialized_example, decoder):
    serialized_example = tf.reshape(tf_serialized_example, [])
    # Return Tensordict
    keys = decoder.list_items()
    tensors = decoder.decode(serialized_example, items=keys)
    tensor_dict = dict(zip(keys, tensors))
    return tensor_dict


def build(input_reader_config):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')

    reader_config = input_reader_config.tf_record_input_reader
    if reader_config is None:
        raise ValueError('input_reader_config must have '
                         '`tf_record_input_reader`.')

    input_record_patterns = reader_config.input_path
    input_record_paths = tf.gfile.Glob(input_record_patterns)
    num_records = len(input_record_paths)
    if num_records == 0:
        raise ValueError('At least one input path must be specified in '
                         '`input_reader_config`.')

    files_dataset = tf.data.Dataset.from_tensor_slices(input_record_paths)
    files_dataset = files_dataset.repeat()
    dataset = files_dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=input_reader_config.num_readers))
    if input_reader_config.shuffle:
        dataset = dataset.shuffle(input_reader_config.shuffle_buffer)

    decoder = _create_tf_example_decoder()
    dataset = dataset.map(
        functools.partial(_process_fn, decoder=decoder),
        num_parallel_calls=input_reader_config.num_parallel_calls)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()
