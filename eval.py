import functools
import os
import six
from google.protobuf import text_format
import tensorflow as tf

from builders import dataset_builder
from builders import preprocessor_builder as preprocessor
from builders import model_builder
from protos import pipeline_pb2


tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('evaluate_all_from_checkpoint', None,
                    'Path to a checkpoint from which to begin running '
                    'evaluation from. All proceeding checkpoints will also '
                    ' be evaluated.')

flags.DEFINE_string('train_dir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.mark_flag_as_required('train_dir')

flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.mark_flag_as_required('eval_dir')

flags.DEFINE_string('config_path', '',
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('config_path')

flags.DEFINE_boolean('image_summaries', False,
                     'Show summaries of eval predictions and save to '
                     'Tensorboard for viewing.')

flags.DEFINE_boolean('verbose', False,
                     'Show streamed mIoU updates during evaluation of '
                     'various checkpoint evaluation runs.')


def get_checkpoints_from_path(initial_checkpoint_path, checkpoint_dir):
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints is None:
        raise ValueError('No checkpoints found in `train_dir`.')
    all_checkpoints = checkpoints.all_model_checkpoint_paths
    # Loop through all checkpoints in the specified dir
    checkpoints_to_evaluate = None
    tf.logging.info('Searching checkpoints in %s', checkpoint_dir)
    for idx, ckpt in enumerate(all_checkpoints):
        print(idx, ' ', str(ckpt))
        if str(ckpt) == FLAGS.evaluate_all_from_checkpoint:
            checkpoints_to_evaluate = all_checkpoints[idx:]
            break
    # We must be able to find the checkpoint specified
    if checkpoints_to_evaluate is None:
        raise ValueError('Checkpoint not found. Exiting.')
    return checkpoints_to_evaluate


def create_evaluation_input(create_input_dict_fn,
                            input_height,
                            input_width,
                            cropped_eval=False):
    input_dict = create_input_dict_fn()
    # We evaluate on a random cropped of the validation set.
    if cropped_eval:
        cropper_fn = functools.partial(preprocessor.random_crop,
                       crop_height=input_height,
                       crop_width=input_width)
        input_dict = preprocessor.preprocess_runner(
                input_dict, func_list=[cropper_fn])
    else:
        padding_fn = functools.partial(preprocessor.pad_to_specific_size,
                        height_to_set=input_height,
                        width_to_set=input_width)
        input_dict = preprocessor.preprocess_runner(
                input_dict, func_list=[padding_fn])
    # Output labels ready for inference
    processed_images = tf.to_float(input_dict[dataset_builder._IMAGE_FIELD])
    processed_labels = tf.to_float(input_dict[dataset_builder._LABEL_FIELD])
    return processed_images, processed_labels


def create_predictions_and_labels(model, create_input_dict_fn,
                                 input_height, input_width, cropped_eval):
    eval_input_pair = create_evaluation_input(
        create_input_dict_fn, input_height, input_width, cropped_eval)
    # Setup a queue for feeding to slim evaluation helpers
    input_queue = prefetch_queue.prefetch_queue(eval_input_pair)
    eval_images, eval_labels = input_queue.dequeue()
    eval_images = tf.expand_dims(eval_images, 0)
    eval_labels = tf.expand_dims(eval_labels, 0)
    # Main predictions
    mean_subtracted_inputs = model.preprocess(eval_images)
    model.provide_groundtruth(eval_labels)
    output_dict = model.predict(mean_subtracted_inputs)
    # Validation loss to fight overfitting
    validation_losses = model.loss(output_dict)
    eval_total_loss =  sum(validation_losses.values())
    # Argmax final outputs to feed to a metric function
    model_scores = output_dict[model.main_class_predictions_key]
    eval_predictions = tf.argmax(model_scores, 3)
    eval_predictions = tf.expand_dims(eval_predictions, -1)

    return eval_predictions, eval_labels, eval_images, eval_total_loss


def eval_segmentation_model(create_model_fn,
                            create_input_fn,
                            input_dimensions,
                            eval_config,
                            train_dir,
                            eval_dir,
                            cropped_evaluation=False,
                            evaluate_all_from_checkpoint=None,
                            image_summaries=False):
    ignore_label = eval_config.ignore_label
    num_classes, segmentation_model = create_model_fn()

    input_height, input_width = input_dimensions
    (predictions_for_eval, labels_for_eval, inputs_summary,
      validation_loss_summary) = create_predictions_and_labels(
                model=segmentation_model,
                create_input_dict_fn=create_input_fn,
                input_height=input_height,
                input_width=input_width,
                cropped_eval=cropped_evaluation)
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
    if FLAGS.verbose:
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
            'InputImage', inputs, family='EvalImages')
        groundtruth_viz = tf.cast(labels*pixel_scaling, tf.uint8)
        tf.summary.image(
            'GroundtruthImage', groundtruth_viz, family='EvalImages')
        predictions_viz = tf.cast(predictions*pixel_scaling, tf.uint8)
        tf.summary.image(
            'PredictionImage', predictions_viz, family='EvalImages')
    summary_op = tf.summary.merge_all()

    tf.logging.info('Evaluating over %d samples...',
                    eval_config.num_examples)

    total_eval_examples = eval_config.num_examples
    if evaluate_all_from_checkpoint is not None:
        checkpoints_to_evaluate = get_checkpoints_from_path(
            evaluate_all_from_checkpoint, train_dir)
        # Run eval on each checkpoint only once. Exit when done.
        for curr_checkpoint in checkpoints_to_evaluate:
            metric_results = slim.evaluation.evaluate_once(
                                master='',
                                checkpoint_path=curr_checkpoint,
                                logdir=eval_dir,
                                num_evals=total_eval_examples,
                                eval_op=eval_op,
                                final_op=value_op,
                                summary_op=summary_op,
                                variables_to_restore=variables_to_restore)
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
                            variables_to_restore=variables_to_restore)
        tf.logging.info('Evaluation over. Eval values: {}'.format(
                        metric_results))


def main(_):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        raise ValueError('`train_dir` must be a valid directory '
                         'containing model checkpoints from training.')
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    eval_config = pipeline_config.eval_config
    input_config = pipeline_config.eval_input_reader
    model_config = pipeline_config.model

    create_input_fn = functools.partial(
        dataset_builder.build,
        input_reader_config=input_config)
    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False)

    eval_input_type = eval_config.eval_input_type
    input_type = eval_input_type.WhichOneof('eval_input_type_oneof')
    if input_type == 'cropped_eval_input':
        cropped_eval_input = eval_input_type.cropped_eval_input
        input_dims = (cropped_eval_input.height,
                      cropped_eval_input.width)
        cropped_evaluation = True
    elif input_type == 'padded_eval_input':
        padded_eval_input = eval_input_type.padded_eval_input
        input_dims = (padded_eval_input.height,
                      padded_eval_input.width)
        cropped_evaluation = False
    else:
        raise ValueError('Must specify an `eval_input_type` for evaluation.')

    eval_segmentation_model(
        create_model_fn,
        create_input_fn,
        input_dims,
        eval_config,
        train_dir=FLAGS.train_dir,
        eval_dir=FLAGS.eval_dir,
        cropped_evaluation=cropped_evaluation,
        evaluate_all_from_checkpoint=FLAGS.evaluate_all_from_checkpoint,
        image_summaries=FLAGS.image_summaries)


if __name__ == '__main__':
    tf.app.run()
