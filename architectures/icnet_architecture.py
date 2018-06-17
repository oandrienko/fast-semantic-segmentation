r"""Contains the definition of the ICNet Semantic Segmentation architecture.

As described in http://arxiv.org/abs/1704.08545.

  ICNet for Real-Time Semantic Segmentation
    on High-Resolution Images
  Hengshuang Zhao, Xiaojuan Qi, Xiaoyong Shen, Jianping Shi, Jiaya Jia

Notes on training:
Training must be done in multiple stages if being done from feature extractor
weights that were trained using ImageNet. The training process incrementally
reduces the number of filters in the architecture to speed up inference time
and memory consumtion.
"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf

import fast_segmentation_model as model

slim = tf.contrib.slim


class ICNetArchitecture(model.FastSegmentationModel):
    """ICNet Architecture definition."""

    def __init__(self,
                is_training,
                model_arg_scope,
                num_classes,
                feature_extractor,
                classification_loss,
                filter_scale,
                use_aux_loss=True,
                main_loss_weight=1,
                second_branch_loss_weight=0,
                first_branch_loss_weight=0,
                batch_norm_decay=0.9997,
                batch_norm_epsilon=1e-5,
                add_summaries=True,
                scope=None):
        super(ICNetArchitecture, self).__init__(num_classes=num_classes)
        self._is_training = is_training
        self._model_arg_scope = model_arg_scope
        self._num_classes = num_classes
        self._feature_extractor = feature_extractor
        self._classification_loss = classification_loss
        self._use_aux_loss = use_aux_loss
        self._main_loss_weight = main_loss_weight
        self._second_branch_loss_weight = second_branch_loss_weight
        self._first_branch_loss_weight = first_branch_loss_weight
        self._filter_scale = filter_scale
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._add_summaries = add_summaries

    @property
    def shared_feature_extractor_scope(self):
        return 'SharedFeatureExtractor'

    def preprocess(self, inputs):
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')

        with tf.variable_scope('Preprocessor'):
            return self._feature_extractor.preprocess(inputs)

    def predict(self, preprocessed_inputs, scope=None):
        """Build main inference pass"""
        with slim.arg_scope(self._model_arg_scope):
            # Feature extraction from arbitrary extractor
            half_res, quarter_res, act = self._extract_shared_features(
                 preprocessed_inputs,
                 scope=self.shared_feature_extractor_scope)
            # Branch specific layers
            pooled_quarter_res = self._first_feature_branch(quarter_res)
            full_res = self._third_feature_branch(preprocessed_inputs)
            # Fusions of branches
            first_fusion, first_aux_logits = self._cascade_feature_fusion(
                pooled_quarter_res, half_res,
                scope="CascadeFeatureFusion_0")
            second_fusion, second_aux_logits = self._cascade_feature_fusion(
                first_fusion, full_res, scope="CascadeFeatureFusion_1")
            # Class class_predictions
            with tf.variable_scope("Predictions"):
                interp_fusion = self._dynamic_interpolation(second_fusion, z_factor=2)
                final_logits = slim.conv2d(interp_fusion, self._num_classes, 1, 1,
                               activation_fn=None, normalizer_fn=None)
            # Outputs with auxilarary loss for training
            prediction_dict = {
                'class_predictions': final_logits}
            if self._use_aux_loss:
                prediction_dict['first_aux_predictions'] = first_aux_logits
                prediction_dict['second_aux_predictions'] = second_aux_logits
            return prediction_dict

    def _extract_shared_features(self, preprocessed_inputs, scope):
        half_scape_inputs = self._dynamic_interpolation(
            preprocessed_inputs, s_factor=2)
        return self._feature_extractor.extract_features(
            half_scape_inputs,
            scope=scope)

    def _first_feature_branch(self, input_features):
        """A suggestion here is to first train the first resolution
        branche without considering other branches. After some M number of
        steps, begin training all branches as normal.
        """
        with tf.variable_scope('PSPModule'):
            _, input_h, input_w, _ = input_features.get_shape()
            full_pool = slim.avg_pool2d(input_features, [input_h, input_w],
                                stride=(input_h, input_w))
            full_pool = tf.image.resize_bilinear(full_pool,
                                size=(input_h, input_w),
                                align_corners=True)
            half_pool = slim.avg_pool2d(input_features,
                                        [input_h//2, input_w//2],
                                stride=(input_h//2, input_w//2))
            half_pool = tf.image.resize_bilinear(half_pool,
                                size=(input_h, input_w),
                                align_corners=True)
            third_pool = slim.avg_pool2d(input_features,
                                        [input_h//3, input_w//3],
                                stride=(input_h//3, input_w//3))
            third_pool = tf.image.resize_bilinear(third_pool,
                                size=(input_h, input_w),
                                align_corners=True)
            forth_pool = slim.avg_pool2d(input_features,
                                        [input_h//6, input_w//6],
                                stride=(input_h//6, input_w//6))
            forth_pool = tf.image.resize_bilinear(forth_pool,
                                size=(input_h, input_w),
                                align_corners=True)
            branch_merge = tf.add_n([input_features, full_pool,
                                     half_pool, third_pool, forth_pool])
            output = slim.conv2d(branch_merge, 512//self._filter_scale, [1, 1],
                                 stride=1, normalizer_fn=slim.batch_norm,
                                 scope='Conv1x1')
            return output

    def _third_feature_branch(self, preprocessed_inputs):
        net = slim.conv2d(preprocessed_inputs, 64//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        net = slim.conv2d(net, 64//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        net = slim.conv2d(net, 128//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        output = slim.conv2d(net, 256//self._filter_scale, [3,3],
                stride=1, normalizer_fn=slim.batch_norm)
        return output

    def _cascade_feature_fusion(self, first_feature_map,
                                second_feature_map, scope):
        with tf.variable_scope(scope):
            upsampled_inputs = self._dynamic_interpolation(first_feature_map, z_factor=2)
            dilated_conv = slim.conv2d(upsampled_inputs, 256//self._filter_scale,
                                      [3, 3], stride=1, rate=2,
                                       normalizer_fn=slim.batch_norm,
                                       activation_fn=None,
                                       scope="DilatedConv")
            conv = slim.conv2d(second_feature_map, 256//self._filter_scale,
                               [1, 1], stride=1,
                               normalizer_fn=slim.batch_norm,
                               activation_fn=None,
                               scope="Conv")
            # Hack to avoid conv shape (1, 129, 257, 256) and
            # dilated_conv shape shape (1, 128, 256, 256) mismatch when
            # evaluating on 1025x2049 cityscapes
            conv = tf.image.resize_bilinear(conv, size=dilated_conv.shape[1:3])
            branch_merge = tf.add_n([conv, dilated_conv])
            output = tf.nn.relu(branch_merge)

            aux_output = slim.conv2d(upsampled_inputs, self._num_classes, 1, 1,
                               activation_fn=None, normalizer_fn=None,
                               scope="AuxOutput")
        return output, aux_output

    def _dynamic_interpolation(self, features_to_upsample, z_factor=1, s_factor=1):
        with tf.variable_scope('Interp'):
            _, input_h, input_w, _ = features_to_upsample.shape
            new_shape = (int(input_h*z_factor/s_factor),
                         int(input_w*z_factor/s_factor))
            return tf.image.resize_bilinear(features_to_upsample, new_shape,
                align_corners=True)

    def loss(self, prediction_dict, scope=None):
        losses_dict = {}
        # TODO: Make this an optional choice. For now only scale
        # down labels like in original paper
        def _resize_labels_to_logits(labels, logits, num_classes):
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, logits.shape[1:3], align_corners=True)
            return scaled_labels
        def _resize_logits_to_labels(logits, labels, num_classes, s_factor=1):
            labels_h, labels_w = labels.shape[1:3]
            new_logits_size = (labels_h//s_factor, labels_w//s_factor)
            scaled_logits = tf.image.resize_bilinear(
                logits, new_logits_size, align_corners=True)
            return scaled_logits

        main_preds = prediction_dict['class_predictions']
        with tf.name_scope('SegmentationLoss'): # 1/4 labels
            main_scaled_labels = _resize_labels_to_logits(
                self._groundtruth_labels, main_preds,
                num_classes=self._num_classes)
            main_loss = self._classification_loss(main_preds,
                                            main_scaled_labels)
            losses_dict['loss'] = (main_loss * self._main_loss_weight)

        if self._use_aux_loss:
            first_aux_preds = prediction_dict['first_aux_predictions']
            second_aux_preds = prediction_dict['second_aux_predictions']
            with tf.name_scope('FirstBranchAuxLoss'): # 1/16 labels
                first_scaled_labels = _resize_labels_to_logits(
                    self._groundtruth_labels, first_aux_preds,
                    num_classes=self._num_classes)
                first_aux_loss = self._classification_loss(first_aux_preds,
                                                        first_scaled_labels)
                losses_dict['first_aux_loss'] = (
                    self._first_branch_loss_weight * first_aux_loss)
            with tf.name_scope('SecondBranchAuxLoss'): # 1/8 labels
                second_scaled_labels = _resize_labels_to_logits(
                    self._groundtruth_labels, second_aux_preds,
                    num_classes=self._num_classes)
                second_aux_loss = self._classification_loss(second_aux_preds,
                                                        second_scaled_labels)
                losses_dict['second_aux_loss'] = (
                    self._second_branch_loss_weight * second_aux_loss)
        return losses_dict

    def restore_map(self,
                    fine_tune_checkpoint_type='segmentation'):
        if fine_tune_checkpoint_type not in ['segmentation', 'classification', 'segmentation-finetune']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
                fine_tune_checkpoint_type))
        if fine_tune_checkpoint_type == 'classification':
            return self._feature_extractor.restore_from_classification_checkpoint_fn(
                self.shared_feature_extractor_scope)

        exclude_list = ['global_step']
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)
        if fine_tune_checkpoint_type == 'segmentation':
            variables_to_restore.append(slim.get_or_create_global_step())
        
        return variables_to_restore


class ICNetFeatureExtractor(object):
    """ICNet Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        self._is_training = is_training
        self._features_stride = features_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

    @abstractmethod
    def preprocess(self, resized_inputs):
        pass

    def extract_features(self, preprocessed_inputs, scope=None):
        """Extracts half resolution features."""
        with tf.variable_scope(
                scope, values=[preprocessed_inputs], reuse=tf.AUTO_REUSE):
            return self._extract_features(preprocessed_inputs, scope)

    @abstractmethod
    def _extract_features(self, preprocessed_inputs, scope):
        pass

    def restore_from_classification_checkpoint_fn(self, scope_name):
        variables_to_restore = {}
        for variable in tf.global_variables():
            if variable.op.name.startswith(scope_name):
                var_name = variable.op.name.replace(scope_name + '/', '')
                variables_to_restore[var_name] = variable
        return variables_to_restore
