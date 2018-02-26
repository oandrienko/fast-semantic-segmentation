"""Resnet V1 ICNet feature extracter interface implementation."""
import tensorflow as tf
from nets import resnet_utils

import dilated_resnet_v1
from architectures import icnet_architecture


slim = tf.contrib.slim


class ICNetResnetV1FeatureExtractor(icnet_architecture.ICNetFeatureExtractor):
    """ICNet feature extractor implementation."""

    _channel_means = [123.68, 116.779, 103.939]

    def __init__(self,
                 architecture,
                 resnet_model,
                 is_training,
                 features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        if features_stride != 8:
            raise ValueError('`features_stride` must be 8 for ICNet,')
        self._architecture = architecture
        self._resnet_model = resnet_model
        super(ICNetResnetV1FeatureExtractor, self).__init__(
            is_training, features_stride, batch_norm_trainable,
            reuse_weights, weight_decay)

    def preprocess(self, raw_inputs):
        channel_means = self._channel_means # RGB VGG ImageNet mean
        return raw_inputs - [[channel_means]]

    def _extract_features(self, preprocessed_inputs, scope):
        half_res_scope = scope + '/%s/block1' % \
                                                        self._architecture
        quarter_res_scope = scope + '/%s/block4' % self._architecture
        with slim.arg_scope(
            resnet_utils.resnet_arg_scope(
                batch_norm_epsilon=1e-5,
                batch_norm_scale=True,
                weight_decay=self._weight_decay)):
            _, activations = self._resnet_model(
                    preprocessed_inputs,
                    num_classes=None,
                    is_training=self._train_batch_norm,
                    global_pool=False,
                    output_stride=self._features_stride)
            half_res_features = activations[half_res_scope]
            quarter_res_features = activations[quarter_res_scope]
            return half_res_features, quarter_res_features, activations


class ICNetDilatedResnet50FeatureExtractor(ICNetResnetV1FeatureExtractor):
    """ICNet Dilated Resnet 50 feature extractor implementation.

    The implementation with dilations contains dilated convolutions in the last
    two blocks of the network. This is how the resnet backbone is
    implemented in the original ICNet paper.
    """

    def __init__(self,
                 is_training,
                 features_stride=8,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        super(ICNetDilatedResnet50FeatureExtractor, self).__init__(
            'resnet_v1_50', dilated_resnet_v1.dilated_resnet_v1_50, is_training,
            features_stride, batch_norm_trainable,
            reuse_weights, weight_decay)
