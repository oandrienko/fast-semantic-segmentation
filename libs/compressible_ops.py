r"""Ops compatable with filter pruner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def convolution2d_compressible(inputs,
                               num_outputs,
                               kernel_size,
                               stride=1,
                               compression_ratio=1.0,  # The additional arg
                               prediction_output=False,
                               **kwargs):
    if prediction_output:
        kwargs['activation_fn'] = None
        kwargs['normalizer_fn'] = None
    num_filter_with_compression = num_outputs // compression_ratio
    return tf.contrib.slim.conv2d(inputs,
                       num_filter_with_compression,
                       kernel_size,
                       stride,
                       **kwargs)


# Export aliases.
compressible_conv2d = convolution2d_compressible  # pylint: disable=C0103
conv2d_compressible = convolution2d_compressible  # pylint: disable=C0103,E0303
conv2d = convolution2d_compressible  # pylint: disable=C0103,E0303
