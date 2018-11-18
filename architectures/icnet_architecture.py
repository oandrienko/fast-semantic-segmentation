r"""ICNet Semantic Segmentation architecture.

As described in http://arxiv.org/abs/1704.08545.

  ICNet for Real-Time Semantic Segmentation
    on High-Resolution Images
  Hengshuang Zhao, Xiaojuan Qi, Xiaoyong Shen, Jianping Shi, Jiaya Jia
"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf

from . import base_model as model

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
                 pretrain_single_branch_mode=False,
                 use_aux_loss=True,
                 main_loss_weight=1.0,
                 first_branch_loss_weight=0.0,
                 second_branch_loss_weight=0.0,
                 upsample_train_logits=False,
                 add_summaries=True,
                 no_add_n_op=False,
                 scope=None):
        super(ICNetArchitecture, self).__init__(num_classes=num_classes)
        self._is_training = is_training
        self._model_arg_scope = model_arg_scope
        self._num_classes = num_classes
        self._feature_extractor = feature_extractor
        self._filter_scale = filter_scale
        self._pretrain_single_branch_mode = pretrain_single_branch_mode
        self._classification_loss = classification_loss
        self._use_aux_loss = use_aux_loss
        self._main_loss_weight = main_loss_weight
        self._first_branch_loss_weight = first_branch_loss_weight
        self._second_branch_loss_weight = second_branch_loss_weight
        self._add_summaries = add_summaries
        self._no_add_n_op = no_add_n_op
        self._upsample_train_logits = upsample_train_logits
        self._output_zoom_factor = (8
            if self._pretrain_single_branch_mode else 4)

    @property
    def main_class_predictions_key(self):
        return 'class_predictions'

    @property
    def first_aux_predictions_key(self):
        return 'first_aux_predictions'

    @property
    def second_aux_predictions_key(self):
        return 'second_aux_predictions'

    @property
    def single_branch_mode_predictions_key(self):
        return 'single_branch_mode_predictions'

    @property
    def main_loss_key(self):
        return 'loss'

    @property
    def first_aux_loss_key(self):
        return 'first_aux_loss'

    @property
    def second_aux_loss_key(self):
        return 'second_aux_loss'

    @property
    def pretrain_single_branch_mode_loss_key(self):
        return 'pretrain_single_branch_mode_loss'

    def preprocess(self, inputs):
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            return self._feature_extractor.preprocess(inputs)

    def _extract_shared_features(self, preprocessed_inputs, scope):
        if not self._pretrain_single_branch_mode:
            extractor_inputs = self._dynamic_interpolation(
                    preprocessed_inputs, s_factor=0.5)
        else:
            extractor_inputs = preprocessed_inputs
        outputs = self._feature_extractor.extract_features(
                extractor_inputs, scope=scope)
        return outputs

    def predict(self, preprocessed_inputs, scope=None):
        """Build main inference pass"""
        with slim.arg_scope(self._model_arg_scope):
            # Feature extraction from arbitrary extractor
            half_res, quarter_res, psp_aux_out = self._extract_shared_features(
                 preprocessed_inputs,
                 scope=self.shared_feature_extractor_scope)
            # Branch specific layers
            pooled_quarter_res = self._icnet_pspmodule(quarter_res)

            # We enable the option to train using only the main resolution
            # branch which includes the PSPNet based module
            if not self._pretrain_single_branch_mode:
                # Full resolution branch
                full_res = self._third_feature_branch(preprocessed_inputs)
                # Fusions of all branches using CFF module
                first_fusion, first_aux_logits = self._cascade_feature_fusion(
                        pooled_quarter_res, half_res,
                        scope='CascadeFeatureFusion_0')
                (second_fusion,
                  second_aux_logits) = self._cascade_feature_fusion(
                        first_fusion, full_res,
                        scope='CascadeFeatureFusion_1')
                final_logits = self._dynamic_interpolation(
                        second_fusion, z_factor=2.0)
            else:
                final_logits = pooled_quarter_res

            # Class class_predictions
            with tf.variable_scope('Predictions'):
                output_name_scope = ('pretrain'
                    if self._pretrain_single_branch_mode else 'postrain')
                predictions = slim.conv2d(final_logits, self._num_classes,
                                1, 1, activation_fn=None, normalizer_fn=None,
                                scope=output_name_scope)
                if not self._is_training: # evaluation output
                    predictions = self._dynamic_interpolation(predictions,
                                            z_factor=self._output_zoom_factor)
            # Main output used in both pretrain and regular train mode
            prediction_dict = {
                self.main_class_predictions_key: predictions}
            # Auxilarary loss for training all three ICNet branches
            if self._is_training and self._use_aux_loss:
                if self._pretrain_single_branch_mode:
                    with tf.variable_scope('AuxPredictions'):
                        psp_aux_out = slim.conv2d(psp_aux_out,
                                self._num_classes, 1, 1,
                                activation_fn=None, normalizer_fn=None)
                        prediction_dict[
                        self.single_branch_mode_predictions_key] = psp_aux_out
                else:
                    prediction_dict[
                        self.first_aux_predictions_key] = first_aux_logits
                    prediction_dict[
                        self.second_aux_predictions_key] = second_aux_logits
            return prediction_dict

    def _icnet_pspmodule(self, input_features):
        """A suggestion here is to first train the first resolution
        branche without considering other branches. After some M number of
        steps, begin training all branches as normal.
        """
        with tf.variable_scope('FastPSPModule'):
            (input_n, input_h, input_w,
              input_c) = input_features.get_shape().as_list()
            full_pool = slim.avg_pool2d(input_features,
                    [input_h, input_w], stride=[input_h, input_w])
            full_pool = tf.image.resize_bilinear(full_pool,
                    size=(input_h, input_w), align_corners=True)
            half_pool = slim.avg_pool2d(input_features,
                    [input_h/2, input_w/2], stride=[input_h/2, input_w/2])
            half_pool = tf.image.resize_bilinear(half_pool,
                    size=(input_h, input_w), align_corners=True)
            third_pool = slim.avg_pool2d(input_features,
                    [input_h/3, input_w/3], stride=[input_h/3, input_w/3])
            third_pool = tf.image.resize_bilinear(third_pool,
                    size=(input_h, input_w), align_corners=True)
            forth_pool = slim.avg_pool2d(input_features,
                    [input_h/6, input_w/6], stride=[input_h/6, input_w/6])
            forth_pool = tf.image.resize_bilinear(forth_pool,
                    size=(input_h, input_w), align_corners=True)

            if self._no_add_n_op:
                branch_merge = tf.add(input_features, full_pool)
                branch_merge = tf.add(branch_merge, half_pool)
                branch_merge = tf.add(branch_merge, third_pool)
                branch_merge = tf.add(branch_merge, forth_pool)
            else:
                branch_merge = tf.add_n([input_features, full_pool,
                                     half_pool, third_pool, forth_pool])

            output = slim.conv2d(branch_merge,
                    512//self._filter_scale, [1, 1],
                    stride=1, normalizer_fn=slim.batch_norm)
            return output

    def _third_feature_branch(self, preprocessed_inputs):
        conv_0 = slim.conv2d(preprocessed_inputs,
                64//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        conv_1 = slim.conv2d(conv_0,
                64//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        conv_2 = slim.conv2d(conv_1,
                128//self._filter_scale, [3,3],
                stride=2, normalizer_fn=slim.batch_norm)
        output = slim.conv2d(conv_2,
                256//self._filter_scale, [3,3],
                stride=1, normalizer_fn=slim.batch_norm)
        return output

    def _cascade_feature_fusion(self, first_feature_map,
                                second_feature_map, scope):
        aux_output = None
        with tf.variable_scope(scope):
            upsampled_inputs = self._dynamic_interpolation(
                    first_feature_map, z_factor=2.0)
            dilated_conv = slim.conv2d(upsampled_inputs,
                    256//self._filter_scale, [3, 3],
                    stride=1, rate=2, normalizer_fn=slim.batch_norm,
                    activation_fn=None, scope='DilatedConv')
            conv = slim.conv2d(second_feature_map,
                    256//self._filter_scale, [1, 1],
                    stride=1, normalizer_fn=slim.batch_norm,
                    activation_fn=None, scope='Conv')
            dilated_conv = tf.image.resize_bilinear(dilated_conv,
                                                    size=conv.shape[1:3])
            # branch_merge = tf.add_n([conv, dilated_conv])
            branch_merge = tf.add(conv, dilated_conv)
            output = tf.nn.relu(branch_merge)
            # Output aux predictions if aux loss_enableded
            if self._is_training and self._use_aux_loss:
                aux_output = slim.conv2d(upsampled_inputs,
                        self._num_classes, 1, 1,
                        activation_fn=None, normalizer_fn=None,
                        scope='AuxOutput')
        return output, aux_output

    def _dynamic_interpolation(self, features_to_upsample,
                               s_factor=1.0, z_factor=1.0):
        with tf.name_scope('Interp'):
            _, input_h, input_w, _ = features_to_upsample.get_shape().as_list()
            shrink_h = (input_h-1)*s_factor+1
            shrink_w = (input_w-1)*s_factor+1
            zoom_h = shrink_h + (shrink_h-1)*(z_factor-1)
            zoom_w = shrink_w + (shrink_w-1)*(z_factor-1)
            return tf.image.resize_bilinear(features_to_upsample,
                                            size=[int(zoom_h), int(zoom_w)],
                                            align_corners=True)

    def loss(self, prediction_dict, scope=None):
        losses_dict = {}
        # TODO: Make this an optional choice. For now only scale
        # down labels like in original paper
        def _resize_labels_to_logits(labels, logits):
            logits_shape = logits.get_shape().as_list()
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, logits_shape[1:3], align_corners=True)
            return scaled_labels

        main_preds = prediction_dict[self.main_class_predictions_key]
        with tf.name_scope('SegmentationLoss'): # 1/4 labels
            if self._upsample_train_logits:
                main_preds = self._dynamic_interpolation(main_preds,
                            z_factor=self._output_zoom_factor)
            main_scaled_labels = _resize_labels_to_logits(
                self._groundtruth_labels, main_preds)
            main_loss = self._classification_loss(main_preds,
                                            main_scaled_labels)
            losses_dict[
                self.main_loss_key] = (main_loss * self._main_loss_weight)

        if self._is_training and self._use_aux_loss:
            if not self._pretrain_single_branch_mode:
                first_aux_preds = prediction_dict[
                        self.first_aux_predictions_key]
                second_aux_preds = prediction_dict[
                        self.second_aux_predictions_key]
                with tf.name_scope('FirstBranchAuxLoss'): # 1/16 labels
                    if self._upsample_train_logits:
                        first_aux_preds = self._dynamic_interpolation(
                            first_aux_preds,
                            z_factor=self._output_zoom_factor)
                    first_scaled_labels = _resize_labels_to_logits(
                        self._groundtruth_labels, first_aux_preds)
                    first_aux_loss = self._classification_loss(
                        first_aux_preds, first_scaled_labels)
                    losses_dict[self.first_aux_loss_key] = (
                        self._first_branch_loss_weight * first_aux_loss)
                with tf.name_scope('SecondBranchAuxLoss'): # 1/8 labels
                    if self._upsample_train_logits:
                        second_aux_preds = self._dynamic_interpolation(
                            second_aux_preds,
                            z_factor=self._output_zoom_factor)
                    second_scaled_labels = _resize_labels_to_logits(
                        self._groundtruth_labels, second_aux_preds)
                    second_aux_loss = self._classification_loss(
                        second_aux_preds, second_scaled_labels)
                    losses_dict[self.second_aux_loss_key] = (
                        self._second_branch_loss_weight * second_aux_loss)
            else:
                with tf.name_scope('PretrainMainAuxLoss'): # 1/8 labels
                    psp_pretrain_preds = prediction_dict[
                            self.single_branch_mode_predictions_key]
                    psp_aux_scaled_labels = _resize_labels_to_logits(
                            self._groundtruth_labels, psp_pretrain_preds)
                    psp_pretrain_loss = self._classification_loss(
                            psp_pretrain_preds, psp_aux_scaled_labels)
                    losses_dict[self.pretrain_single_branch_mode_loss_key] = (
                            0.4 * psp_pretrain_loss)
        return losses_dict

    def restore_map(self, fine_tune_checkpoint_type='segmentation'):
        """Restore variables for checkpoints correctly"""
        if fine_tune_checkpoint_type not in [
                'segmentation', 'classification', 'segmentation-finetune']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(fine_tune_checkpoint_type))
        if fine_tune_checkpoint_type == 'classification':
            tf.logging.info('Fine-tuning from classification checkpoints.')
            return self._feature_extractor.restore_from_classif_checkpoint_fn(
                self.shared_feature_extractor_scope)
        exclude_list = ['global_step']
        variables_to_restore = slim.get_variables_to_restore(
                                        exclude=exclude_list)
        if fine_tune_checkpoint_type == 'segmentation-finetune':
            tf.logging.info('Fine-tuning from PSPNet based checkpoint.')
            variables_to_restore.append(slim.get_or_create_global_step())
        return variables_to_restore
