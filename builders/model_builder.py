from builders import hyperparams_builder
from builders import losses_builder
from protos import model_pb2

from extractors import icnet_resnet_v1
from architectures import icnet_architecture

# TODO(@oandrien): Add MobileNet feature extractor
ICNET_FEATURE_EXTRACTER = {
    'dilated_resnet50':
        icnet_resnet_v1.ICNetDilatedResnet50FeatureExtractor
}


def _build_icnet_extractor(
        feature_extractor_config, is_training, reuse_weights=None):
    feature_type = feature_extractor_config.type

    if feature_type not in ICNET_FEATURE_EXTRACTER:
        raise ValueError('Unknown ICNet feature_extractor: {}'.format(
            feature_type))

    feature_extractor_class = ICNET_FEATURE_EXTRACTER[
        feature_type]
    return feature_extractor_class(is_training, reuse_weights=reuse_weights)


def _build_icnet_model(icnet_config, is_training, add_summaries):
    num_classes = icnet_config.num_classes
    if not num_classes:
        raise ValueError('"num_classes" must be greater than 0.')

    feature_extractor = _build_icnet_extractor(
      icnet_config.feature_extractor, is_training)

    filter_scale = icnet_config.filter_scale
    if filter_scale > 1 or filter_scale < 0:
        raise ValueError('"filter_scale" must be in the range (0,1].')

    model_arg_scope = hyperparams_builder.build(
        icnet_config.hyperparams, is_training)

    classification_loss = (
        losses_builder.build(icnet_config.loss))
    use_aux_loss = icnet_config.loss.use_aux_loss

    common_kwargs = {
        'is_training': is_training,
        'num_classes': num_classes,
        'model_arg_scope': model_arg_scope,
        'num_classes': num_classes,
        'feature_extractor': feature_extractor,
        'classification_loss': classification_loss,
        'use_aux_loss': use_aux_loss,
        'add_summaries': add_summaries
    }

    if use_aux_loss:
        loss_config = icnet_config.loss
        common_kwargs['main_loss_weight'] = loss_config.main_loss_weight
        common_kwargs[
            'second_branch_loss_weight'] = loss_config.second_branch_loss_weight
        common_kwargs[
            'first_branch_loss_weight'] = loss_config.first_branch_loss_weight

    return icnet_architecture.ICNetArchitecture(
        filter_scale=filter_scale,
        **common_kwargs)


def build(model_config, is_training, add_summaries=True):
    if not isinstance(model_config, model_pb2.FastSegmentationModel):
        raise ValueError('model_config not of type model_pb2.FastSegmentationModel.')
    model = model_config.WhichOneof('model')

    if model == 'icnet':
        return _build_icnet_model(model_config.icnet, is_training, add_summaries)

    raise ValueError('Unknown model: {}'.format(model))
