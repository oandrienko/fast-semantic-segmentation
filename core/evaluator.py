import functools
import os
import six
from google.protobuf import text_format
import tensorflow as tf

from builders import dataset_builder
from builders import model_builder
from protos import pipeline_pb2


slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue


def encode_image_array_as_png_str(image):
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def create_predictions_and_labels(model, create_input_dict_fn,
                                 input_height, input_width):
    input_dict = create_input_dict_fn()
    images = tf.to_float(input_dict[
                    dataset_builder._IMAGE_FIELD])
    images.set_shape((input_height, input_width, 3))
    labels = tf.to_float(input_dict[
                    dataset_builder._LABEL_FIELD])
    labels.set_shape((input_height, input_width, 1))

    input_queue = prefetch_queue.prefetch_queue([images, labels])
    input_list = input_queue.dequeue()

    out_labels = inputs = tf.expand_dims(input_list[1], 0)

    inputs = model.preprocess(input_list[0])
    inputs = tf.expand_dims(inputs, 0)
    output_dict = model.predict(inputs)
    outputs = output_dict['class_predictions']
    outputs = tf.argmax(outputs, 3)
    outputs = tf.expand_dims(outputs, -1)
    out_outputs = tf.image.resize_bilinear(outputs,
        size=(input_height, input_width),
        align_corners=True)
    out_images = tf.expand_dims(input_list[0], 0)
    return out_outputs, out_labels, out_images


def eval_segmentation_model_once(eval_config, input_config, model_config,
                                 train_dir,
                                 eval_dir,
                                 checkpoint_path,
                                 master=''):

    create_input_fn = functools.partial(
        dataset_builder.build,
        input_reader_config=input_config)
    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False)
    ignore_label = eval_config.ignore_label

    num_classes, segmentation_model = create_model_fn()
    predictions, labels, inputs = create_predictions_and_labels(
        model=segmentation_model,
        create_input_dict_fn=create_input_fn,
        input_height=eval_config.fixed_height,
        input_width=eval_config.fixed_width)

    # Gather variables from training
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    flattened_predictions = tf.reshape(predictions, shape=[-1])
    flattened_labels = tf.reshape(labels, shape=[-1])

    validity_mask = tf.equal(
                flattened_labels, ignore_label)
    neg_validity_mask = tf.not_equal(
                flattened_labels, ignore_label)
    eval_labels = tf.where(validity_mask, tf.zeros_like(
        flattened_labels), flattened_labels)
    eval_predictions = flattened_predictions

    # Define the evaluation metric.
    metric_map = {}
    predictions_tag="mIoU"
    metric_map[predictions_tag] = tf.contrib.metrics.streaming_mean_iou(
                        eval_predictions, eval_labels, num_classes,
                        weights=tf.to_float(neg_validity_mask))
    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))
    for metric_name, metric_value in six.iteritems(metrics_to_values):
        tf.summary.scalar(metric_name,  metric_value)
    eval_op = list(metrics_to_updates.values())

    tf.logging.info('Evaluating over %d samples...',
                    eval_config.num_examples)

    summary_op = tf.summary.merge_all()

    final_result = slim.evaluation.evaluate_once(
                    master=master,
                    checkpoint_path=checkpoint_path,
                    logdir=eval_dir,
                    num_evals=eval_config.num_examples,
                    eval_op=eval_op,
                    final_op=metric_map[predictions_tag],
                    summary_op=summary_op,
                    variables_to_restore=variables_to_restore)

    return final_result
