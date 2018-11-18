"""MobileNet V2 feature extracter interface implementation."""
import tensorflow as tf

from third_party import mobilenet_v2
from architectures import pspnet_architecture


slim = tf.contrib.slim


class PSPNetICNetMobilenetV2FeatureExtractor(
                pspnet_architecture.PSPNetFeatureExtractor):
    """ICNet feature extractor implementation."""

    _channel_means = [127.5, 127.5, 127.5]

    def __init__(self,
                 architecture,
                 mobilenet_model,
                 is_training,
                 filter_scale,
                 features_stride,
                 mid_downsample=False,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        if features_stride != 8:
            raise ValueError('`features_stride` must be 8 '
                             'for ICNet and PSPNet.')
        self._filter_scale = filter_scale
        self._architecture = architecture
        self._mobilenet_model = mobilenet_model
        self._mid_downsample = mid_downsample
        # Mobilenet specific options
        self._depth_multiplier = 1.0
        super(PSPNetICNetMobilenetV2FeatureExtractor, self).__init__(
            is_training, features_stride, batch_norm_trainable,
            reuse_weights, weight_decay)

    def preprocess(self, raw_inputs):
        channel_means = self._channel_means # We normalize between [-1, 1]
        return (raw_inputs - [[channel_means]]) / [[channel_means]]

    def _extract_features(self, preprocessed_inputs, scope):
        half_res_scope = 'layer_5' # expanded_conv_3
        quarter_res_scope = 'layer_18' # expanded_conv_16
        psp_aux_scope = 'layer_8' # expanded_conv_6

        conv_defs = mobilenet_v2.make_conv_defs(
                filter_scale=self._filter_scale,
                mid_downsample=self._mid_downsample)
        with slim.arg_scope(
            mobilenet_v2.training_scope(
                is_training=self._is_training,
                weight_decay=self._weight_decay)):
            logits, activations = mobilenet_v2.mobilenet_base(
                preprocessed_inputs,
                conv_defs=conv_defs,
                depth_multiplier=self._depth_multiplier,
                min_depth=(8 if self._depth_multiplier == 1.0 else 1),
                divisible_by=(8 if self._depth_multiplier == 1.0 else 1),
                output_stride=self._features_stride,
                final_endpoint='layer_18')

            half_res_features = activations[half_res_scope]
            quarter_res_features = activations[quarter_res_scope]
            psp_aux_features = activations[psp_aux_scope]
            return half_res_features, quarter_res_features, psp_aux_features

class PSPNetICNetMobilenetFeatureExtractor(
        PSPNetICNetMobilenetV2FeatureExtractor):
    """ICNet Dilated Resnet 50 feature extractor implementation.

    The implementation with dilations contains dilated convolutions in the last
    two blocks of the network. This is how the resnet backbone is
    implemented in the original ICNet paper.
    """

    def __init__(self,
                 is_training,
                 filter_scale=1.0,
                 mid_downsample=False,
                 features_stride=8,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        super(PSPNetICNetMobilenetFeatureExtractor, self).__init__(
            'MobilenetV2', mobilenet_v2.mobilenet_base, is_training,
            filter_scale, features_stride, mid_downsample, batch_norm_trainable,
            reuse_weights, weight_decay)
