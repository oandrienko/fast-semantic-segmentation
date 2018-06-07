import functools
import os
import six
from google.protobuf import text_format
import tensorflow as tf

from builders import dataset_builder
from builders import model_builder
from protos import pipeline_pb2


tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', '',
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.mark_flag_as_required('logdir')

flags.DEFINE_string('eval_dir', '',
                    'Directory to write eval summaries to.')
flags.mark_flag_as_required('eval_dir')

flags.DEFINE_string('config_path', '',
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('config_path')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How often do run evaluation loop in seconds. Defaults '
                     'to 5 minutes (300s).')


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


def main(_):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
    eval_config = pipeline_config.eval_config
    input_config = pipeline_config.eval_input_reader
    model_config = pipeline_config.model

    ## SOME VALIDATION HERE

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

    # Image summaries
    global_step = tf.train.get_global_step()
    pixel_scaling = max(1, 255 // num_classes)
    tf.summary.image(
            "InputImage", inputs, family="Images")
    groundtruth_viz = tf.cast(labels*pixel_scaling, tf.uint8)
    tf.summary.image(
            "GroundtruthImage", groundtruth_viz, family="Images")
    predictions_viz = tf.cast(predictions*pixel_scaling, tf.uint8)
    tf.summary.image(
            "PredictionImage", predictions_viz, family="Images")

    # Define the evaluation metric.
    metric_map = {}
    predictions_tag="mIoU"
    metric_map[predictions_tag] = tf.metrics.mean_iou(
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

    # eval every 5 minutes
    max_number_of_evaluations = None
    if eval_config.max_evals:
        max_number_of_evaluations = eval_config.max_evals
    slim.evaluation.evaluation_loop(
        eval_op=eval_op,
        summary_op=summary_op,
        max_number_of_evaluations=max_number_of_evaluations,
        variables_to_restore=variables_to_restore,
        master='',
        checkpoint_dir=FLAGS.logdir,
        logdir=FLAGS.eval_dir,
        num_evals=eval_config.num_examples,
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    tf.app.run()
