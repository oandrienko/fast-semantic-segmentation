import functools
import json
import os
import time
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import gradients
from google.protobuf import text_format

# from deployment import model_deploy
from third_party import model_deploy
from third_party import memory_saving_gradients
from third_party import mem_util

from builders import model_builder
from builders import dataset_builder
from builders import preprocessor_builder
from builders import optimizer_builder
from protos import pipeline_pb2


tf.logging.set_verbosity(tf.logging.INFO)

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

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

flags.DEFINE_integer('max_checkpoints_to_keep', 50, # might want to cut this down
                     'Number of checkpoints to keep in the `logdir`.')

# Debug flag

flags.DEFINE_boolean('gradient_checkpointing', False, '')

flags.DEFINE_boolean('show_memory', False, '')

flags.DEFINE_boolean('test_image_summaries', False, '')

flags.DEFINE_boolean('tmp_icnet_branch_summaries', False, 'temp flag')

flags.DEFINE_boolean('tmp_psp_pretrain_summaries', False, 'temp flag')



#################### TEMP, TO FIT FOR TRAINING ######################

 # Monkey patch tf.gradients
if FLAGS.gradient_checkpointing:
    def gradients_memory(ys, xs, grad_ys=None, **kwargs):
        return memory_saving_gradients.gradients(
            ys, xs, grad_ys, checkpoints='collection', **kwargs)
    gradients.__dict__["gradients"] = gradients_memory

######################################################################



def create_training_input(create_input_fn,
                          preprocess_fn,
                          batch_size,
                          batch_queue_capacity,
                          batch_queue_threads,
                          prefetch_queue_capacity):

    tensor_dict = create_input_fn()

    def cast_and_reshape(tensor_dict, dicy_key):
        items = tensor_dict[dicy_key]
        float_images = tf.to_float(items)
        tensor_dict[dicy_key] = float_images
        return tensor_dict

    tensor_dict = cast_and_reshape(tensor_dict,
                    dataset_builder._IMAGE_FIELD)

    if preprocess_fn is not None:
        preprocessor = preprocess_fn()
        tensor_dict = preprocessor(tensor_dict)

    batched_tensors = tf.train.batch(tensor_dict,
        batch_size=batch_size, num_threads=batch_queue_threads,
        capacity=batch_queue_capacity, dynamic_pad=True)

    return prefetch_queue.prefetch_queue(batched_tensors,
        capacity=prefetch_queue_capacity,
        dynamic_pad=False)


def create_training_model_losses(input_queue, create_model_fn, train_config):

    _, segmentation_model = create_model_fn()

    # Optional quantization
    if train_config.quantize_with_delay:
        tf.logging.info('Adding quantization nodes to training graph...')
        tf.contrib.quantize.create_training_graph(
            quant_delay=train_config.quantize_with_delay)

    read_data_list = input_queue.dequeue()
    def extract_images_and_targets(read_data):
        images = read_data[dataset_builder._IMAGE_FIELD]
        labels = read_data[dataset_builder._LABEL_FIELD]
        return (images, labels)
    (images, labels) = zip(*map(extract_images_and_targets, [read_data_list]))

    # Incase we need to do zero centering, we do it here
    preprocessed_images = []
    for image in images:
        resized_image = segmentation_model.preprocess(image)
        preprocessed_images.append(resized_image)
    images = tf.concat(preprocessed_images, 0, name="Inputs")

    segmentation_model.provide_groundtruth(labels[0])
    prediction_dict = segmentation_model.predict(images)

    if FLAGS.gradient_checkpointing:
        tf.logging.info('Adding gradient checkpoints to `checkpoints` collection')
        graph = tf.get_default_graph()
        checkpoint_list = segmentation_model.gradient_checkpointing_nodes()
        for checkpoint_node_name in checkpoint_list:
            node = graph.get_tensor_by_name(checkpoint_node_name)
            tf.add_to_collection('checkpoints', node)

    # Gather main and aux losses here to single collection
    losses_dict = segmentation_model.loss(prediction_dict)
    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)


def train_segmentation_model(create_model_fn,
                             create_input_fn,
                             train_config,
                             master,
                             task,
                             is_chief,
                             startup_delay_steps,
                             train_dir,
                             num_clones,
                             num_worker_replicas,
                             num_ps_tasks,
                             clone_on_cpu,
                             replica_id,
                             num_replicas,
                             max_checkpoints_to_keep,
                             save_interval_secs,
                             image_summaries):
    """Create an instance of the FastSegmentationModel"""
    _, segmentation_model = create_model_fn()
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=num_worker_replicas,
        num_ps_tasks=num_ps_tasks)
    startup_delay_steps = task * startup_delay_steps

    per_clone_batch_size = train_config.batch_size // num_clones

    preprocess_fn = None
    if train_config.preprocessor_step:
        preprocess_fn = functools.partial(
            preprocessor_builder.build,
            preprocessor_config_list=train_config.preprocessor_step)

    with tf.Graph().as_default():

        with tf.device(deploy_config.variables_device()): # CPU of common ps server
            global_step = tf.train.get_or_create_global_step()

        with tf.device(deploy_config.inputs_device()): # CPU of each worker
            input_queue = create_training_input(
                create_input_fn,
                preprocess_fn,
                per_clone_batch_size,
                batch_queue_capacity=train_config.batch_queue_capacity,
                batch_queue_threads=train_config.num_batch_queue_threads,
                prefetch_queue_capacity=train_config.prefetch_queue_capacity)

        # Create the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            # Note: it is assumed that any loss created by `model_fn`
            # is collected at the tf.GraphKeys.LOSSES collection.
            model_fn = functools.partial(create_training_model_losses,
                                    create_model_fn=create_model_fn,
                                    train_config=train_config)
            clones = model_deploy.create_clones(deploy_config,
                model_fn, [input_queue])
            first_clone_scope = deploy_config.clone_scope(0)

            # Attempt to sync BN updates across all GPU's in a tower. Caution
            # since this is very slow. Might not be needed
            # update_ops = []
            # for idx in range(num_clones):
            #     nth_clone_sope = deploy_config.clone_scope(0)
            #     update_ops.extend(tf.get_collection(tf.GraphKeys.UPDATE_OPS,
            #                               nth_clone_sope))

            # Gather updates from first GPU only
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                           first_clone_scope)

        # Init variable to collect summeries
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('Losses/%s' % loss.op.name, loss))

        with tf.device(deploy_config.optimizer_device()): # CPU of each worker
            (training_optimizer,
              optimizer_summary_vars) = optimizer_builder.build(
                train_config.optimizer)
            for var in optimizer_summary_vars:
                summaries.add(
                    tf.summary.scalar(var.op.name, var, family='LearningRate'))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Fine tune from classification or segmentation checkpoints
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if train_config.fine_tune_checkpoint:
            if not train_config.fine_tune_checkpoint_type:
                raise ValueError('Must specify `fine_tune_checkpoint_type`.')

            tf.logging.info('Initializing %s model from path: %s',
                train_config.fine_tune_checkpoint_type,
                train_config.fine_tune_checkpoint)

            variables_to_restore = segmentation_model.restore_map(
              fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type)

            init_fn = slim.assign_from_checkpoint_fn(
                        train_config.fine_tune_checkpoint,
                        variables_to_restore,
                        ignore_missing_vars=True)

            if train_config.freeze_fine_tune_backbone:
                non_frozen_vars = []
                for var in trainable_vars:
                    if var.op.name not in variables_to_restore.values():
                        non_frozen_vars.append()
                import pdb; pdb.set_trace()
                trainable_vars = non_frozen_vars
        else:
            tf.logging.info('Not initializing the model from a checkpoint.')

        # TODO(@oandrien): we might want to add gradient multiplier here
        # for the last layer if we have trouble with training
        with tf.device(deploy_config.optimizer_device()): # CPU of common ps server
            reg_losses = (None if train_config.add_regularization_loss
                               else [])
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, training_optimizer,
                regularization_losses=reg_losses)
            total_loss = tf.check_numerics(total_loss,
                                          'LossTensor is inf or nan.')
            summaries.add(
                tf.summary.scalar('Losses/TotalLoss', total_loss))

            grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                        global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                train_op = tf.identity(total_loss, name='train_op')

##############################################################################################################
        # TEMPORARY...
        if image_summaries:
            graph = tf.get_default_graph()
            pixel_scaling = max(1, 255 // 19)
            summ_first_clone_scope = (first_clone_scope + '/'
                if first_clone_scope else '')
            main_in = graph.get_tensor_by_name(
                    '%sfifo_queue_Dequeue:1' % summ_first_clone_scope)
            main_gt = graph.get_tensor_by_name(
                    '%sSegmentationLoss/ResizeNearestNeighbor:0' % summ_first_clone_scope)
            summaries.add(
              tf.summary.image('VerifyTrainImageInput/Inputs', main_in))
            main_gt = tf.cast(main_gt * pixel_scaling, tf.uint8)
            summaries.add(
              tf.summary.image('VerifyTrainImageMain/Groundtruths', main_gt))

            if FLAGS.tmp_icnet_branch_summaries:
                main_out = graph.get_tensor_by_name(
                    '%sPredictions/Conv/BiasAdd:0' % summ_first_clone_scope)
                main_out = tf.expand_dims(tf.argmax(main_out, 3), -1)
                main_out = tf.cast(main_out * pixel_scaling, tf.uint8)
                summaries.add(tf.summary.image('VerifyTrainImageMain/Predictions', main_out))

                aux_out_0 = graph.get_tensor_by_name(
                        '%sCascadeFeatureFusion_0/AuxOutput/BiasAdd:0' % summ_first_clone_scope)
                aux_gt_0 = graph.get_tensor_by_name(
                        '%sFirstBranchAuxLoss/ResizeNearestNeighbor:0' % summ_first_clone_scope)
                aux_out_1 = graph.get_tensor_by_name(
                        '%sCascadeFeatureFusion_1/AuxOutput/BiasAdd:0' % summ_first_clone_scope)
                aux_gt_1 = graph.get_tensor_by_name(
                        '%sSecondBranchAuxLoss/ResizeNearestNeighbor:0' % summ_first_clone_scope)
                aux_out_0 = tf.expand_dims(tf.argmax(aux_out_0, 3), -1)
                aux_out_0 = tf.cast(aux_out_0 * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImageFirstBranchAux/Predictions', aux_out_0))
                aux_gt_0 = tf.cast(aux_gt_0 * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImageFirstBranchAux/Groundtruths', aux_gt_0))

                aux_out_1 = tf.expand_dims(tf.argmax(aux_out_1, 3), -1)
                aux_out_1 = tf.cast(aux_out_1 * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImageSecondBranchAux/Predictions', aux_out_1))
                aux_gt_1 = tf.cast(aux_gt_1 * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImageSecondBranchAux/Groundtruths', aux_gt_1))

            if FLAGS.tmp_psp_pretrain_summaries:
                main_out = graph.get_tensor_by_name(
                    '%sPredictions/PretrainConv/BiasAdd:0' % summ_first_clone_scope)
                main_out = tf.expand_dims(tf.argmax(main_out, 3), -1)
                main_out = tf.cast(main_out * pixel_scaling, tf.uint8)
                summaries.add(tf.summary.image('VerifyTrainImageMain/Predictions', main_out))

                aux_out = graph.get_tensor_by_name(
                        '%sAuxPredictions/Conv/BiasAdd:0' % summ_first_clone_scope)
                aux_gt = graph.get_tensor_by_name(
                        '%sPretrainMainAuxLoss/ResizeNearestNeighbor:0' % summ_first_clone_scope)
                aux_out = tf.expand_dims(tf.argmax(aux_out, 3), -1)
                aux_out = tf.cast(aux_out * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImagePretrainMainAux/Predictions', aux_out))
                aux_gt = tf.cast(aux_gt * pixel_scaling, tf.uint8)
                summaries.add(
                  tf.summary.image('VerifyTrainImagePretrainMainAux/Groundtruths', aux_gt))

##############################################################################################################


        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))

        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)

        # Save checkpoints regularly.
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

        # HACK to see memory usage.
        # TODO: Clean up, pretty messy.
        def train_step_mem(sess, train_op, global_step, train_step_kwargs):
            start_time = time.time()
            run_metadata = tf.RunMetadata()
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=options,
                                        run_metadata=run_metadata)
            time_elapsed = time.time() - start_time

            if 'should_log' in train_step_kwargs:
                if sess.run(train_step_kwargs['should_log']):
                    logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                        np_global_step, total_loss, time_elapsed)

            if FLAGS.show_memory:
                mem_use = mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
                logging.info('Memory used: %.2f MB',(mem_use))

            if 'should_stop' in train_step_kwargs:
                should_stop = sess.run(train_step_kwargs['should_stop'])
            else:
                should_stop = False

            return total_loss, should_stop

        # Main training loop
        slim.learning.train(
            train_op,
            train_step_fn=train_step_mem,
            logdir=train_dir,
            master=master,
            is_chief=is_chief,
            session_config=session_config,
            number_of_steps=train_config.num_steps,
            startup_delay_steps=startup_delay_steps,
            init_fn=init_fn,
            init_op=init_op,
            summary_op=summary_op,
            save_summaries_secs=120,
            save_interval_secs=save_interval_secs,
            saver=saver)


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
        image_summaries=FLAGS.test_image_summaries)


if __name__ == '__main__':
    tf.app.run()
