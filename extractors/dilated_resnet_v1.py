# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Resnet v1 model variants.

Code branched out from slim/nets/resnet_v1.py, and please refer to it for
more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_utils
from tensorflow.contrib.slim.nets import resnet_v1


slim = tf.contrib.slim


class DownSampleBlock(
  collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """ """


@slim.add_arg_scope
def downsample(inputs, s_factor, stride=1, rate=1, scope=None):
  """"""
  with tf.variable_scope(scope, 'Interp', [inputs]) as sc:
    _, input_h, input_w, _ = inputs.get_shape().as_list()
    shrink_h = (input_h-1)*s_factor+1
    shrink_w = (input_w-1)*s_factor+1
    return tf.image.resize_bilinear(inputs,
                                    [int(shrink_h), int(shrink_w)],
                                    align_corners=True)


@slim.add_arg_scope
def bottleneck(inputs,
               depth,
               depth_bottleneck,
               stride,
               overide_rate=1,
               rate=1,
               outputs_collections=None,
               scope=None,
               use_bounded_activations=False):
  """Bottleneck residual unit variant with BN after convolutions.

  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for
  its definition. Note that we use here the bottleneck variant which has an
  extra bottleneck layer.

  When putting together two consecutive ResNet blocks that use this unit, one
  should use stride = 2 in the last unit of the first block.

  Args:
    inputs: A tensor of size [batch, height, width, channels].
    depth: The depth of the ResNet unit output.
    depth_bottleneck: The depth of the bottleneck layers.
    stride: The ResNet unit's stride. Determines the amount of downsampling of
      the units output compared to its input.
    unit_rate: An integer, overiding rate for atrous convolution.
    rate: An integer, rate for atrous convolution.
    outputs_collections: Collection to add the ResNet unit output.
    scope: Optional variable_scope.
    use_bounded_activations: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.

  Returns:
    The ResNet unit's output.
  """
  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv2d(
          inputs,
          depth, [1, 1],
          stride=stride,
          activation_fn=tf.nn.relu6 if use_bounded_activations else None,
          scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1,
                           scope='conv1')
    residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                        rate=rate*overide_rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                           activation_fn=None, scope='conv3')

    if use_bounded_activations:
      # Use clip_by_value to simulate bandpass activation.
      residual = tf.clip_by_value(residual, -6.0, 6.0)
      output = tf.nn.relu6(shortcut + residual)
    else:
      output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet_v1_block(scope, base_depth, num_units, stride, rate=1):
  """Helper function for creating a resnet_v1 bottleneck block.

  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.

  Returns:
    A resnet_v1 bottleneck block.
  """
  minor_depth = int(base_depth)
  major_depth = int(base_depth * 4)
  return resnet_utils.Block(scope, bottleneck, [{
      'depth': major_depth,
      'depth_bottleneck': minor_depth,
      'stride': 1,
      'overide_rate': rate
  }] * (num_units - 1) + [{
      'depth': major_depth,
      'depth_bottleneck': minor_depth,
      'stride': stride,
      'overide_rate': rate
  }])


def resnet_v1_downsample_block(scope, factor):
  """ """
  return DownSampleBlock(scope, downsample, [{
    's_factor': factor
  }])


def resnet_v1(inputs,
              blocks,
              filter_scale=1.0,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              reuse=None,
              scope=None):
  with tf.variable_scope(
      scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope(
        [slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense],
        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          net = resnet_utils.conv2d_same(net, 64//filter_scale, 7, stride=2, scope='conv1')
          net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
        net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
        if global_pool:
          # Global average pooling.
          net = math_ops.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
        if num_classes is not None:
          net = slim.conv2d(
              net,
              num_classes, [1, 1],
              activation_fn=None,
              normalizer_fn=None,
              scope='logits')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = slim.utils.convert_collection_to_dict(
          end_points_collection)
        if num_classes is not None:
          end_points['predictions'] = slim.softmax(
              net, scope='predictions')
        return net, end_points
resnet_v1.default_image_size = 224


def dilated_resnet_v1_50(inputs,
                         filter_scale=1.0,
                         mid_downsample=False,
                         num_classes=None,
                         is_training=True,
                         global_pool=True,
                         output_stride=None,
                         reuse=None,
                         scope='resnet_v1_50'):
  """Resnet v1 50 variant with dilations.

  This variant modifies the first convolution layer of ResNet-v1-50. In
  particular, it changes the original several of the last ResNet blocks
  to include dilated convolutions as used in ICNet.
  """

  blocks = [
    resnet_v1_block('block1', base_depth=64/filter_scale,
                    num_units=3, stride=2)]

  if mid_downsample:
    blocks.append(
      resnet_v1_downsample_block('downsample_block', factor=0.5))

  blocks += [
    resnet_v1_block('block2', base_depth=128/filter_scale,
                    num_units=4, stride=2),
    resnet_v1_block('block3', base_depth=256/filter_scale,
                    num_units=6, stride=2, rate=2),
    resnet_v1_block('block4', base_depth=512/filter_scale,
                    num_units=3, stride=1, rate=4)]

  return resnet_v1(inputs, blocks, filter_scale, num_classes, is_training,
                   global_pool=global_pool,output_stride=output_stride,
                   include_root_block=True, reuse=reuse, scope=scope)
