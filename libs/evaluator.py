from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six
import tensorflow as tf

from builders import dataset_builder
from builders import preprocessor_builder as preprocessor


slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue


def create_evaluation_input(create_input_dict_fn,
                            input_height,
                            input_width,
                            cropped_eval=False):
    input_dict = create_input_dict_fn()
    if cropped_eval:
        # We evaluate on a random cropped of the validation set.
        cropper_fn = functools.partial(preprocessor.random_crop,
                       crop_height=input_height,
                       crop_width=input_width)
        output_dict = preprocessor.preprocess_runner(
                input_dict, func_list=[cropper_fn])
        processed_labels = tf.to_float(
            output_dict[dataset_builder._LABEL_FIELD])
    else:
        # Here we only pad input image, then we shrink back the prediction
        padding_fn = functools.partial(preprocessor.pad_to_specific_size,
                        height_to_set=input_height,
                        width_to_set=input_width)
        output_dict = preprocessor.preprocess_runner(
                input_dict, skip_labels=True, func_list=[padding_fn])
        processed_labels = tf.to_float(input_dict[dataset_builder._LABEL_FIELD])
    processed_images = tf.to_float(output_dict[dataset_builder._IMAGE_FIELD])
    return processed_images, processed_labels


def create_predictions_and_labels(model, create_input_dict_fn,
                                 input_height, input_width, cropped_eval,
                                 eval_dir=None):
    eval_input_pair = create_evaluation_input(
        create_input_dict_fn, input_height, input_width, cropped_eval)
    # Setup a queue for feeding to slim evaluation helpers
    input_queue = prefetch_queue.prefetch_queue(eval_input_pair)
    eval_images, eval_labels = input_queue.dequeue()
    eval_labels = tf.expand_dims(eval_labels, 0)
    eval_images = tf.expand_dims(eval_images, 0)
    # Main predictions
    mean_subtracted_inputs = model.preprocess(eval_images)
    model.provide_groundtruth(eval_labels)
    output_dict = model.predict(mean_subtracted_inputs)

    # Awkward fix from preprocessing step - we resize back down to label shape
    if not cropped_eval:
        eval_labels_shape = eval_labels.get_shape().as_list()
        padded_predictions = output_dict[model.main_class_predictions_key]
        padded_predictions = tf.image.resize_bilinear(padded_predictions,
            size=eval_labels_shape[1:3],
            align_corners=True)
        output_dict[model.main_class_predictions_key] = padded_predictions

    # Output graph def for pruning
    if eval_dir is not None:
        graph_def = tf.get_default_graph().as_graph_def()
        pred_graph_def_path = os.path.join(eval_dir, "eval_graph.pbtxt")
        f = tf.gfile.FastGFile(pred_graph_def_path, "w")
        f.write(str(graph_def))
    # Validation loss to fight overfitting
    validation_losses = model.loss(output_dict)
    eval_total_loss =  sum(validation_losses.values())
    # Argmax final outputs to feed to a metric function
    model_scores = output_dict[model.main_class_predictions_key]
    eval_predictions = tf.argmax(model_scores, 3)
    eval_predictions = tf.expand_dims(eval_predictions, -1)

    return eval_predictions, eval_labels, eval_images, eval_total_loss


def eval_segmentation_model_once(checkpoint_path,
                                 create_model_fn,
                                 create_input_fn,
                                 input_dimensions,
                                 eval_config,
                                 eval_dir,
                                 cropped_evaluation=False,
                                 image_summaries=False,
                                 verbose=False,
                                 sess_config=None):
    return eval_segmentation_model(
        create_model_fn,
        create_input_fn,
        input_dimensions,
        eval_config,
        train_dir=None,
        eval_dir=eval_dir,
        cropped_evaluation=cropped_evaluation,
        evaluate_single_checkpoint=checkpoint_path,
        image_summaries=image_summaries,
        verbose=verbose,
        sess_config=sess_config)


def eval_segmentation_model(create_model_fn,
                            create_input_fn,
                            input_dimensions,
                            eval_config,
                            train_dir,
                            eval_dir,
                            cropped_evaluation=False,
                            evaluate_single_checkpoint=None,
                            image_summaries=False,
                            verbose=False,
                            sess_config=None):
    ignore_label = eval_config.ignore_label
    num_classes, segmentation_model = create_model_fn()

    input_height, input_width = input_dimensions
    (predictions_for_eval, labels_for_eval, inputs_summary,
      validation_loss_summary) = create_predictions_and_labels(
                model=segmentation_model,
                create_input_dict_fn=create_input_fn,
                input_height=input_height,
                input_width=input_width,
                cropped_eval=cropped_evaluation,
                eval_dir=eval_dir)
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    # Prepare inputs to metric calculation steps
    flattened_predictions = tf.reshape(predictions_for_eval, shape=[-1])
    flattened_labels = tf.reshape(labels_for_eval, shape=[-1])
    validity_mask = tf.equal(flattened_labels, ignore_label)
    neg_validity_mask = tf.not_equal(flattened_labels, ignore_label)
    eval_labels = tf.where(validity_mask, tf.zeros_like(
            flattened_labels), flattened_labels)
    # Calculate metrics from predictions
    metric_map = {}
    predictions_tag='EvalMetrics/mIoU'
    value_op, update_op = tf.contrib.metrics.streaming_mean_iou(
                        flattened_predictions, eval_labels, num_classes,
                        weights=tf.to_float(neg_validity_mask))
    # Print updates if verbosity is requested
    if verbose:
        update_op = tf.Print(update_op, [value_op], predictions_tag)
    # TODO: Extend the metrics tuple if needed in the future
    metric_map[predictions_tag] = (value_op, update_op)
    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))
    for metric_name, metric_value in six.iteritems(metrics_to_values):
        tf.summary.scalar(metric_name,  metric_value)
    eval_op = list(metrics_to_updates.values())

    # Summaries for Tensorboard
    if validation_loss_summary is not None:
        tf.summary.scalar("Losses/EvalValidationLoss",
            validation_loss_summary)
    # Image summaries if requested
    if image_summaries:
        pixel_scaling = max(1, 255 // num_classes)
        tf.summary.image(
            'InputImage', inputs_summary, family='EvalImages')
        groundtruth_viz = tf.cast(labels_for_eval*pixel_scaling, tf.uint8)
        tf.summary.image(
            'GroundtruthImage', groundtruth_viz, family='EvalImages')
        predictions_viz = tf.cast(predictions_for_eval*pixel_scaling, tf.uint8)
        tf.summary.image(
            'PredictionImage', predictions_viz, family='EvalImages')
    summary_op = tf.summary.merge_all()

    tf.logging.info('Evaluating over %d samples...',
                    eval_config.num_examples)

    total_eval_examples = eval_config.num_examples
    if evaluate_single_checkpoint:
        curr_checkpoint = evaluate_single_checkpoint
        metric_results = slim.evaluation.evaluate_once(
                            master='',
                            checkpoint_path=curr_checkpoint,
                            logdir=eval_dir,
                            num_evals=total_eval_examples,
                            eval_op=eval_op,
                            final_op=value_op,
                            summary_op=summary_op,
                            variables_to_restore=variables_to_restore,
                            session_config=sess_config)
        tf.logging.info('Evaluation of `{}` over. Eval values: {}'.format(
                    curr_checkpoint, metric_results))
    else:
        metric_results = slim.evaluation.evaluation_loop(
                            master='',
                            checkpoint_dir=train_dir,
                            logdir=eval_dir,
                            num_evals=total_eval_examples,
                            eval_op=eval_op,
                            final_op=value_op,
                            summary_op=summary_op,
                            variables_to_restore=variables_to_restore,
                            session_config=sess_config)
        tf.logging.info('Evaluation over. Eval values: {}'.format(
                        metric_results))

    return metric_results
