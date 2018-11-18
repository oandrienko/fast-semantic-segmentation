from builders import hyperparams_builder
from builders import losses_builder
from protos import model_pb2

from extractors import pspnet_icnet_resnet_v1
from extractors import pspnet_icnet_mobilenet_v2
from architectures import pspnet_architecture
from architectures import icnet_architecture


PSPNET_ICNET_FEATURE_EXTRACTER = {
    'dilated_resnet50':
        pspnet_icnet_resnet_v1.PSPNetICNetDilatedResnet50FeatureExtractor,
    'dilated_mobilenet':
        pspnet_icnet_mobilenet_v2.PSPNetICNetMobilenetFeatureExtractor
}


def _build_pspnet_icnet_extractor(
        feature_extractor_config, filter_scale, is_training,
        mid_downsample=False, reuse_weights=None):
    feature_type = feature_extractor_config.type

    if feature_type not in PSPNET_ICNET_FEATURE_EXTRACTER:
        raise ValueError('Unknown ICNet feature_extractor: {}'.format(
            feature_type))

    feature_extractor_class = PSPNET_ICNET_FEATURE_EXTRACTER[
        feature_type]
    return feature_extractor_class(is_training,
                                   batch_norm_trainable=is_training,
                                   filter_scale=filter_scale,
                                   mid_downsample=mid_downsample,
                                   reuse_weights=reuse_weights)

def _build_pspnet_icnet_model(model_config, is_training, add_summaries,
                              build_baseline_psp=False):
    num_classes = model_config.num_classes
    if not num_classes:
        raise ValueError('"num_classes" must be greater than 0.')

    in_filter_scale = model_config.filter_scale
    if in_filter_scale > 1 or in_filter_scale < 0:
        raise ValueError('"filter_scale" must be in the range (0,1].')
    filter_scale = 1.0 / in_filter_scale

    should_downsample_extractor = False
    if not build_baseline_psp:
        pretrain_single_branch_mode = model_config.pretrain_single_branch_mode
        should_downsample_extractor = not pretrain_single_branch_mode

    feature_extractor = _build_pspnet_icnet_extractor(
            model_config.feature_extractor, filter_scale, is_training,
            mid_downsample=should_downsample_extractor)

    model_arg_scope = hyperparams_builder.build(model_config.hyperparams,
                                                is_training)

    loss_config = model_config.loss
    classification_loss = (
            losses_builder.build(loss_config))
    use_aux_loss = loss_config.use_auxiliary_loss

    common_kwargs = {
        'is_training': is_training,
        'num_classes': num_classes,
        'model_arg_scope': model_arg_scope,
        'num_classes': num_classes,
        'feature_extractor': feature_extractor,
        'classification_loss': classification_loss,
        'use_aux_loss': use_aux_loss,
        'upsample_train_logits': loss_config.upsample_logits,
        'add_summaries': add_summaries
    }

    if not build_baseline_psp:
        if use_aux_loss:
            common_kwargs['main_loss_weight'] = (
                model_config.main_branch_loss_weight)
            common_kwargs['second_branch_loss_weight'] = (
                model_config.second_branch_loss_weight)
            common_kwargs['first_branch_loss_weight'] = (
                model_config.first_branch_loss_weight)
        common_kwargs['no_add_n_op'] = (
            model_config.mobile_ops_only)
        model = (num_classes, icnet_architecture.ICNetArchitecture(
            filter_scale=filter_scale,
            pretrain_single_branch_mode=pretrain_single_branch_mode,
            **common_kwargs))
    else:
        if use_aux_loss:
            # TODO: remove hardcoded values here
            common_kwargs['main_loss_weight'] = 1.0
            common_kwargs['aux_loss_weight'] = 0.4
        model = (num_classes, pspnet_architecture.PSPNetArchitecture(
            filter_scale=filter_scale,
            **common_kwargs))
    return model


def build(model_config, is_training, add_summaries=True):
    if not isinstance(model_config, model_pb2.FastSegmentationModel):
        raise ValueError('model_config not of type '
                         'model_pb2.FastSegmentationModel.')

    model = model_config.WhichOneof('model')
    if model == 'pspnet':
        return _build_pspnet_icnet_model(
            model_config.pspnet, is_training, add_summaries,
            build_baseline_psp=True)
    elif model == 'icnet':
        return _build_pspnet_icnet_model(
            model_config.icnet, is_training, add_summaries)

    raise ValueError('Unknown model: {}'.format(model))
