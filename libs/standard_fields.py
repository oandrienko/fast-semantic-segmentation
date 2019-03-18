r"""Commonly used keys."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class TFRecordFields(object):
    """Keys for parsing TFRecords."""
    image_encoded = 'image/encoded'
    image_filename = 'image/filename'
    image_format = 'image/format'
    image_height = 'image/height'
    image_width = 'image/width'
    image_channels = 'image/channels'
    segmentation_class_encoded = 'image/segmentation/class/encoded'
    segmentation_class_format = 'image/segmentation/class/format'


class GroundtruthFields(object):
    """Keys for groudtruth dicts."""
    input_image = 'input_image'
    input_image_path = 'input_image_path'
    input_image_height = 'input_image_height'
    input_image_width = 'input_image_width'
    output_mask = 'label_mask'
