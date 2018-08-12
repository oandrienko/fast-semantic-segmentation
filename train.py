r"""Main Training script for ICNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from google.protobuf import text_format

from builders import model_builder
from builders import dataset_builder
from protos import pipeline_pb2
from libs.trainer import train_segmentation_model


tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags

FLAGS = flags.FLAGS

# Distributed training settings

flags.DEFINE_integer('num_clones', 1,
                     'Number of model clones to deploy to each worker replica.'
                     'This should be greater than one if you want to use '
                     'multiple GPUs located on a single machine.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1,
                     'Number of worker replicas. This typically corresponds '
                     'to the number of machines you are training on. Note '
                     'that the training will be done asynchronously.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker. It is '
                     'reccomended to use num_ps_tasks=num_replicas/2.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID. Should increment per worker '
                     'replica added to achieve between graph replication.')

# Training configuration settings

flags.DEFINE_string('config_path', '',
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('config_path')

flags.DEFINE_string('logdir', '',
                    'Directory to save the checkpoints and training summaries.')
flags.mark_flag_as_required('logdir')

flags.DEFINE_integer('save_interval_secs', 600, # default to 5 min
                     'Time between successive saves of a checkpoint in secs.')

flags.DEFINE_integer('max_checkpoints_to_keep', 15, # might want to cut this down
                     'Number of checkpoints to keep in the `logdir`.')

# Debug flag

flags.DEFINE_boolean('image_summaries', False, '')


def main(_):
    tf.gfile.MakeDirs(FLAGS.logdir)
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)

    model_config = pipeline_config.model
    train_config = pipeline_config.train_config
    input_config = pipeline_config.train_input_reader

    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    create_input_fn = functools.partial(
        dataset_builder.build,
        input_reader_config=input_config)

    is_chief = (FLAGS.task == 0)

    train_segmentation_model(
        create_model_fn,
        create_input_fn,
        train_config,
        master=FLAGS.master,
        task=FLAGS.task,
        is_chief=is_chief,
        startup_delay_steps=FLAGS.startup_delay_steps,
        train_dir=FLAGS.logdir,
        num_clones=FLAGS.num_clones,
        num_worker_replicas=FLAGS.num_replicas,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.num_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks,
        max_checkpoints_to_keep=FLAGS.max_checkpoints_to_keep,
        save_interval_secs=FLAGS.save_interval_secs,
        image_summaries=FLAGS.image_summaries)


if __name__ == '__main__':
    tf.app.run()
