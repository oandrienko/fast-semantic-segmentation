import tensorflow as tf
from protos import hyperparams_pb2

slim = tf.contrib.slim


def _build_regularizer(regularizer):
    regularizer_oneof = regularizer.WhichOneof('regularizer_oneof')
    if  regularizer_oneof == 'l1_regularizer':
        return slim.l1_regularizer(scale=float(
            regularizer.l1_regularizer.weight))
    if regularizer_oneof == 'l2_regularizer':
        return slim.l2_regularizer(scale=float(
            regularizer.l2_regularizer.weight))
    raise ValueError('Unknown regularizer function: {}'.format(
        regularizer_oneof))


def _build_initializer(initializer):
    initializer_oneof = initializer.WhichOneof('initializer_oneof')
    if initializer_oneof == 'truncated_normal_initializer':
        return tf.truncated_normal_initializer(
            mean=initializer.truncated_normal_initializer.mean,
            stddev=initializer.truncated_normal_initializer.stddev)
    if initializer_oneof == 'variance_scaling_initializer':
        enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.
                            DESCRIPTOR.enum_types_by_name['Mode'])
        mode = enum_descriptor.values_by_number[initializer.
                                            variance_scaling_initializer.
                                            mode].name
        return slim.variance_scaling_initializer(
            factor=initializer.variance_scaling_initializer.factor,
            mode=mode,
            uniform=initializer.variance_scaling_initializer.uniform)
    raise ValueError('Unknown initializer function: {}'.format(
        initializer_oneof))


def build(hyperparams_config, is_training):
    batch_norm = None
    batch_norm_params = None
    if hyperparams_config.HasField('batch_norm'):
        batch_norm = hyperparams_config.batch_norm
        batch_norm_params = {
            'decay': batch_norm.decay,
            'center': batch_norm.center,
            'scale': batch_norm.scale,
            'epsilon': batch_norm.epsilon,
            'is_training': is_training and batch_norm.train}

    affected_ops = [slim.conv2d,
        slim.separable_conv2d, slim.conv2d_transpose]
    with slim.arg_scope(
        affected_ops,
        weights_regularizer=_build_regularizer(
            hyperparams_config.regularizer),
        weights_initializer=_build_initializer(
            hyperparams_config.initializer),
        activation_fn=tf.nn.relu6,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):

        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
            return sc
