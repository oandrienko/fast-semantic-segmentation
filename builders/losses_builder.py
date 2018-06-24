from functools import partial
import tensorflow as tf
from protos import losses_pb2


def _softmax_classification_loss(predictions, labels, ignore_label):
    flattened_labels = tf.reshape(labels, shape=[-1])
    num_classes = predictions.get_shape().as_list()[-1]
    predictions = tf.reshape(predictions, [-1, num_classes])

    one_hot_target = tf.contrib.slim.one_hot_encoding(
                            tf.cast(flattened_labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    not_ignore_mask = tf.to_float(
                tf.not_equal(flattened_labels, ignore_label))

    return tf.losses.softmax_cross_entropy(
                    one_hot_target,
                    logits=tf.to_float(predictions),
                    weights=not_ignore_mask)

def build(loss_config):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')

    loss_type = loss_config.classification_loss.WhichOneof('loss_type')
    if loss_type == 'softmax':
        return partial(_softmax_classification_loss,
            ignore_label=loss_config.ignore_label)

    raise ValueError('Empty loss config.')
