import os
import functools
import tensorflow as tf

from protos import input_reader_pb2

slim = tf.contrib.slim

tfexample_decoder = slim.tfexample_decoder

dataset = slim.dataset

dataset_data_provider = slim.dataset_data_provider

_DATASET_SHUFFLE_SEED = 7

_IMAGE_FIELD            = 'image'
_IMAGE_NAME_FIELD       = 'image_name'
_HEIGHT_FIELD           = 'height'
_WIDTH_FIELD            = 'width'
_LABEL_FIELD           = 'labels_class'

_ITEMS_TO_DESCRIPTIONS = {
    'image':        ('A color image of varying height and width.'),
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}


def _create_tf_example_decoder():

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    input_image = tfexample_decoder.Image(
        image_key='image/encoded',
        format_key='image/format',
        shape=(1024, 2048, 3), # CITYSCAPES SPECIFIC
        channels=3)
    ground_truth_image = tfexample_decoder.Image(
        image_key='image/segmentation/class/encoded',
        format_key='image/segmentation/class/format',
        shape=(1024, 2048, 1), # CITYSCAPES SPECIFIC
        channels=1)

    items_to_handlers = {
        _IMAGE_FIELD: input_image,
        _IMAGE_NAME_FIELD: tfexample_decoder.Tensor('image/filename'),
        _HEIGHT_FIELD: tfexample_decoder.Tensor('image/height'),
        _WIDTH_FIELD: tfexample_decoder.Tensor('image/width'),
        _LABEL_FIELD: ground_truth_image,
    }

    return tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


def build(input_reader_config):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')

    reader_config = input_reader_config.tf_record_input_reader
    if reader_config is None:
        raise ValueError('input_reader_config must have '
                             '`tf_record_input_reader`.')

    if not reader_config.input_path or \
            not os.path.isfile(reader_config.input_path[0]):
        raise ValueError('At least one input path must be specified in '
                         '`input_reader_config`.')

    decoder = _create_tf_example_decoder()

    train_dataset = dataset.Dataset(
        data_sources=reader_config.input_path[:],
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=input_reader_config.num_examples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

    provider = dataset_data_provider.DatasetDataProvider(
        train_dataset,
        num_readers=input_reader_config.num_readers,
        num_epochs=(input_reader_config.num_epochs
            if input_reader_config.num_epochs else None),
        shuffle=input_reader_config.shuffle,
        seed=_DATASET_SHUFFLE_SEED)

    (image, image_name, height, width, label) = provider.get([_IMAGE_FIELD,
        _IMAGE_NAME_FIELD, _HEIGHT_FIELD, _WIDTH_FIELD, _LABEL_FIELD])

    return {
        _IMAGE_FIELD: image,
        _IMAGE_NAME_FIELD: image_name,
        _HEIGHT_FIELD: height,
        _WIDTH_FIELD: width,
        _LABEL_FIELD: label
    }
