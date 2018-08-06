from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from builders import dataset_builder
from builders import preprocessor_builder as preprocessor


slim = tf.contrib.slim


def _map_to_colored_labels(label, shape, num_classes, color_map):
    color_map_constant_mat = []
    for color in color_map:
        color_map_constant_mat.append(list(color))
    color_table = tf.constant(color_map_constant_mat, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int32)
    onehot_labels = tf.one_hot(label, depth=num_classes)
    onehot_labels = tf.reshape(onehot_labels, (-1, num_classes))
    colored_label = tf.matmul(onehot_labels, color_table)
    colored_label = tf.reshape(colored_label, (1, shape[1], shape[2], 3))
    return colored_label

def _get_outputs_from_inputs(input_tensors, model,
                             output_collection_name):
    # models expect a batch dimension
    if len(input_tensors.get_shape()) < 4:
        input_tensors = tf.expand_dims(input_tensors, axis=0)
    # model expect a floating point input
    inputs = tf.to_float(input_tensors)
    # build model
    preprocessed_inputs = model.preprocess(inputs)
    outputs_dict = model.predict(preprocessed_inputs)
    output_tensors = outputs_dict[model.main_class_predictions_key]
    prediction_tensor = tf.argmax(output_tensors, 3)
    # name tensor to make inference with frozen weights easier
    final_op = tf.identity(prediction_tensor,
        name=output_collection_name)
    return final_op


def _image_tensor_input_placeholder(input_shape=None, pad_to_shape=None):
    if input_shape is None:
        input_shape = (None, None, None, 3)
    placeholder_tensor = tf.placeholder(
        dtype=tf.uint8, shape=input_shape, name='inputs')
    if pad_to_shape is not None:
        input_tensor = tf.image.pad_to_bounding_box(placeholder_tensor,
            0, 0, pad_to_shape[0], pad_to_shape[1])
    else:
        input_tensor = placeholder_tensor
    return placeholder_tensor, input_tensor


def deploy_segmentation_inference_graph(model, input_shape,
                                        output_collection_name,
                                        pad_to_shape=None,
                                        num_classes=None,
                                        label_color_map=None):
    (placeholder_tensor,
      model_input) = _image_tensor_input_placeholder(input_shape, pad_to_shape)
    outputs = _get_outputs_from_inputs(
        input_tensors=model_input,
        model=model,
        output_collection_name=output_collection_name)
    if label_color_map is not None:
        output_shape = outputs.get_shape().as_list()
        outputs = _map_to_colored_labels(outputs,
                output_shape, num_classes, label_color_map)
    tf.train.get_or_create_global_step()
    return outputs, placeholder_tensor
