r"""Preprocessing step for inptut images"""
import functools
import tensorflow as tf

from protos import preprocessor_pb2
from builders import dataset_builder

RESIZE_METHOD_MAP = {
    preprocessor_pb2.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}

_RANDOM_SCALE_STEP_KEY = 'RANDOM_SCALE_STEP'

_IMAGE_CROP_KEY = 'IMAGE_CROP_STEP'

_IMAGE_SCALE_KEY = 'IMAGE_SCALE_KEY'

_IMAGE_HORIZONTAL_FLIP_KEY = 'IMAGE_HORIZONTAL_FLIP_STEP'

_RANDOM_PREPROCESSOR_SEED = 7


def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache):
    if preprocess_vars_cache is not None:
        var = preprocess_vars_cache.get(function_id)
        if var is None:
            var = generator_func()
        preprocess_vars_cache.update({ function_id: var })
    else:
        var = generator_func()
    return var


def set_fixed_image_size(images,
                         labels,
                         height_to_set,
                         width_to_set,
                         images_channel_dim=3,
                         labels_channel_dim=1,
                         preprocess_vars_cache=None):
    with tf.name_scope('DimensionInput', values=[images, labels]):
        fixed_input_tensor_shape = (
            height_to_set, width_to_set, images_channel_dim)
        images.set_shape(fixed_input_tensor_shape)
        fixed_label_tensor_shape = (
            height_to_set, width_to_set, labels_channel_dim)
        labels.set_shape(fixed_label_tensor_shape)
        return images, labels

def pad_to_specific_size(images,
                         labels,
                         height_to_set,
                         width_to_set,
                         images_channel_dim=3,
                         labels_channel_dim=1,
                         preprocess_vars_cache=None):
    with tf.name_scope('PadInput', values=[images, labels]):
        fixed_input_tensor_shape = (
            height_to_set, width_to_set, images_channel_dim)
        padded_images = tf.image.pad_to_bounding_box(
                            images, 0, 0, height_to_set, width_to_set)
        padded_images.set_shape(fixed_input_tensor_shape)
        fixed_label_tensor_shape = (
            height_to_set, width_to_set, labels_channel_dim)
        padded_labels = None
        if labels is not None:
            padded_labels = tf.image.pad_to_bounding_box(
                                labels, 0, 0, height_to_set, width_to_set)
            padded_labels.set_shape(fixed_label_tensor_shape)
        return padded_images, padded_labels


def _compute_new_static_size(image, min_dimension, max_dimension):
    """Compute new static shape for resize_to_range method."""
    image_shape = image.get_shape().as_list()
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    num_channels = image_shape[2]
    orig_min_dim = min(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    large_scale_factor = min_dimension / float(orig_min_dim)
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]
        new_size = large_size
        if max(large_size) > max_dimension:
          new_size = small_size
    else:
        new_size = large_size
    return tf.constant(new_size + [num_channels])


def resize_to_range(image,
                    label=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True,
                    pad_to_max_dimension=False):
    with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
        if image.get_shape().is_fully_defined():
            new_size = _compute_new_static_size(image,
                                    min_dimension, max_dimension)
        else:
            new_size = _compute_new_dynamic_size(image,
                                    min_dimension, max_dimension)
        new_size = _compute_new_dynamic_size(item,
            min_dimension, max_dimension)
        new_image = tf.image.resize_bilinear(image,
                            new_size[:-1], align_corners=True)
        new_label = tf.image.resize_nearest_neighbor(labels,
                                new_size[:-1], align_corners=True)
    return (new_image, new_label)


def random_scale(images,
                 labels,
                 min_scale_ratio=0.5,
                 max_scale_ratio=2.0,
                 pad_to_dims=None,
                 seed=_RANDOM_PREPROCESSOR_SEED,
                 preprocess_vars_cache=None):
    with tf.name_scope('RandomScale', values=[images, labels]):
        image_height, image_width, _ = images.get_shape().as_list()

        generator_func = functools.partial(
            tf.random_uniform, [],
            minval=min_scale_ratio, maxval=max_scale_ratio,
            dtype=tf.float32, seed=seed)
        size_coef = _get_or_create_preprocess_rand_vars(
            generator_func, _IMAGE_SCALE_KEY,
            preprocess_vars_cache)

        image_newysize = tf.to_int32(
                    tf.multiply(tf.to_float(image_height), size_coef))
        image_newxsize = tf.to_int32(
                    tf.multiply(tf.to_float(image_width), size_coef))
        new_shape = (image_newysize, image_newxsize)

        # Must be 4D tensor for resize ops
        images = tf.expand_dims(images, 0)
        labels = tf.expand_dims(labels, 0)
        scaled_images = tf.image.resize_bilinear(
                                images, new_shape, align_corners=True)
        scaled_labels = tf.image.resize_nearest_neighbor(
                                labels, new_shape, align_corners=True)
        if pad_to_dims is not None:
            crop_height, crop_width = pad_to_dims
            target_height = (image_newysize +
                                tf.maximum(crop_height - image_newysize, 0))
            target_width = (image_newxsize +
                                tf.maximum(crop_width - image_newxsize, 0))
            scaled_images = tf.image.pad_to_bounding_box(
                scaled_images, 0, 0, target_height, target_width)
            scaled_labels = tf.image.pad_to_bounding_box(
                scaled_labels, 0, 0, target_height, target_width)
        output_images = tf.squeeze(scaled_images, [0])
        output_labels = tf.squeeze(scaled_labels, [0])
        return output_images, output_labels


def random_crop(images, labels,
                crop_height, crop_width,
                images_channel_dim=3,
                labels_channel_dim=1,
                preprocess_vars_cache=None):

    def _apply_random_crop(inputs, offsets, crop_shape):
        sliced_inputs = tf.slice(inputs, offsets, crop_shape)
        out_inputs = tf.reshape(sliced_inputs, crop_shape)
        return out_inputs

    with tf.name_scope('RandomCropImage', values=[images, labels]):
        images_shape = tf.shape(images)
        images_height = images_shape[0]
        images_width = images_shape[1]

        max_offset_height = tf.reshape(images_height-crop_height+1, [])
        max_offset_width = tf.reshape(images_width-crop_width+1, [])

        generator_func_height = functools.partial(
            tf.random_uniform,
            shape=[], maxval=max_offset_height, dtype=tf.int32)
        generator_func_width = functools.partial(
            tf.random_uniform,
            shape=[], maxval=max_offset_width, dtype=tf.int32)

        offset_height = _get_or_create_preprocess_rand_vars(
            generator_func_height,
            _IMAGE_CROP_KEY+'_0',
            preprocess_vars_cache)
        offset_width = _get_or_create_preprocess_rand_vars(
            generator_func_width,
            _IMAGE_CROP_KEY+'_1',
            preprocess_vars_cache)

        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
        crop_shape_images = tf.stack(
            [crop_height, crop_width, images_channel_dim])
        crop_shape_labels = tf.stack(
            [crop_height, crop_width, labels_channel_dim])

        cropped_images = _apply_random_crop(images, offsets, crop_shape_images)
        cropped_labels = _apply_random_crop(labels, offsets, crop_shape_labels)

        # Must set shape here or in the set shape preprocessor step
        # when dealing with ICNet
        if images_channel_dim and labels_channel_dim:
            cropped_images.set_shape((crop_height, crop_width,
                                     images_channel_dim))
            cropped_labels.set_shape((crop_height, crop_width,
                                     labels_channel_dim))

        return cropped_images, cropped_labels


def random_horizontal_flip(images,
                           labels,
                           seed=_RANDOM_PREPROCESSOR_SEED,
                           preprocess_vars_cache=None):

    def _flip_image(item):
        flipped_item = tf.image.flip_left_right(item)
        return flipped_item

    with tf.name_scope('RandomHorizontalFlip', values=[images, labels]):
        generator_func = functools.partial(
                            tf.random_uniform, [], seed=seed)
        do_a_flip_random = _get_or_create_preprocess_rand_vars(
            generator_func, _IMAGE_HORIZONTAL_FLIP_KEY,
            preprocess_vars_cache)
        do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

        flipped_images = tf.cond(do_a_flip_random,
            lambda: _flip_image(images), lambda: images)
        flipped_labels = tf.cond(do_a_flip_random,
                lambda: _flip_image(labels), lambda: labels)
        return flipped_images, flipped_labels


def preprocess_runner(tensor_dict, func_list, skip_labels=False, preprocess_vars_cache=None):
    if dataset_builder._IMAGE_FIELD not in tensor_dict \
                or dataset_builder._LABEL_FIELD not in tensor_dict:
        raise ValueError('"tensor_dict" must have both image'
                         'and label fields')
    for item_key in [dataset_builder._IMAGE_FIELD,
                     dataset_builder._LABEL_FIELD]:
        items = tensor_dict[item_key]
        if len(items.get_shape()) != 3:
            raise ValueError('Images or Labels in tensor_dict should be rank 4')
        tensor_dict[item_key] = items

    if preprocess_vars_cache is None:
        preprocess_vars_cache = {}

    images = tf.to_float(tensor_dict[dataset_builder._IMAGE_FIELD])
    images_shape = tf.shape(images)
    # For now, we skip labels preprocessing for eval only, since we
    # do whole image evaluation
    # TODO: Fix this so it doesn't break for training
    labels = None
    if not skip_labels:
        labels = tf.to_float(tensor_dict[dataset_builder._LABEL_FIELD])

    # Apple proprocessor functions
    for preprocessor_step_func in func_list:
        images, labels = preprocessor_step_func(images=images, labels=labels,
                        preprocess_vars_cache=preprocess_vars_cache)

    output_dict = {}
    output_dict[dataset_builder._IMAGE_FIELD] = images
    output_dict[dataset_builder._LABEL_FIELD] = labels
    return output_dict


def build(preprocessor_config_list):
    proprocessor_func_list = []

    for preprocessor_step_config in preprocessor_config_list:
        step_type = preprocessor_step_config.WhichOneof('preprocessing_step')

        # Fixed image width and height for PSP module
        if step_type == 'set_fixed_image_size':
            config = preprocessor_step_config.set_fixed_image_size
            dimension_image_fn = functools.partial(
                set_fixed_image_size,
                height_to_set=config.fixed_height,
                width_to_set=confi.fixed_width,
                images_channel_dim=config.images_channel_dim,
                labels_channel_dim=config.labels_channel_dim)
            proprocessor_func_list.append(dimension_image_fn)

        # Resize the image and keep the aspect_ratio
        if step_type == 'aspect_ratio_image_resize':
            config = preprocessor_step_config.aspect_ratio_image_resize
            if not (config.min_dimension <= config.max_dimension):
                raise ValueError('min_dimension > max_dimension')
            method = RESIZE_METHOD_MAP[config.resize_method]
            image_resizer_fn = functools.partial(
                resize_to_range,
                min_dimension=config.min_dimension,
                max_dimension=config.max_dimension,
                pad_to_max_dimension=config.pad_to_max_dimension)
            proprocessor_func_list.append(image_resizer_fn)

        # Randomly Scale the image
        if step_type == 'random_image_scale':
            config = preprocessor_step_config.random_image_scale
            if not (config.max_scale_ratio >= config.min_scale_ratio):
                raise ValueError('min_scale_ratio > max_scale_ratio')

            pad_to_dims = None
            for cfg in preprocessor_config_list:
                step_t = cfg.WhichOneof('preprocessing_step')
                if step_t == 'random_image_crop':
                    dim = cfg.random_image_crop
                    pad_to_dims = (dim.crop_height, dim.crop_width)

            image_scale_fn = functools.partial(
                random_scale,
                pad_to_dims=pad_to_dims,
                min_scale_ratio=config.min_scale_ratio,
                max_scale_ratio=config.max_scale_ratio)
            proprocessor_func_list.append(image_scale_fn)

        # Randomly crop the image
        if step_type == 'random_image_crop':
            config = preprocessor_step_config.random_image_crop
            image_crop_fn = functools.partial(
                random_crop,
                crop_height=config.crop_height,
                crop_width=config.crop_width,
                images_channel_dim=config.images_channel_dim,
                labels_channel_dim=config.labels_channel_dim)
            proprocessor_func_list.append(image_crop_fn)

        # Random Flips and Rotations
        if step_type == 'random_horizontal_flip':
            config = preprocessor_step_config.random_horizontal_flip
            image_horizontal_flip_fn = functools.partial(
                random_horizontal_flip)
            proprocessor_func_list.append(image_horizontal_flip_fn)

        if len(proprocessor_func_list) <= 0 and \
            len(preprocessor_config_list) > 0:
            raise ValueError('Unknown preprocessing step.')

    preprocessor = functools.partial(
        preprocess_runner,
        func_list=proprocessor_func_list)

    return preprocessor
