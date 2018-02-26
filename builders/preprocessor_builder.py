""" Preprocessing step for inptut images

1) Resize Image
2) Randomly scale input and label
3) Pad image for cropping
3) Zero Centering with FE mean
4) Randomly crop the image and label
5) Randomly left right flip the image
"""
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

_IMAGE_VERTICAL_FLIP_KEY = 'IMAGE_VERTICAL_FLIP_STEP'

_IMAGE_HORIZONTAL_FLIP_KEY = '_MAGE_HORIZONTAL_FLIP_STEP'

_RANDOM_PREPROCESSOR_SEED = 7


def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache,
                                        key=''):
    if preprocess_vars_cache is not None:
        var = preprocess_vars_cache.get(function_id, key)
        if var is None:
            var = generator_func()
        preprocess_vars_cache.update(function_id, key, var)
    else:
        var = generator_func()
    return var


def _compute_new_static_size(image, min_dimension, max_dimension):
    orig_height, orig_width, num_channels = image.get_shape().as_list()
    orig_min_dim = min(orig_height, orig_width)

    large_scale_factor = min_dimension / float(orig_min_dim)
    large_height = int(round(orig_height * large_scale_factor))
    large_width = int(round(orig_width * large_scale_factor))
    large_size = [large_height, large_width]

    new_size = large_size
    if max_dimension:
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / float(orig_max_dim)
        small_height = int(round(orig_height * small_scale_factor))
        small_width = int(round(orig_width * small_scale_factor))
        small_size = [small_height, small_width]

        if max(large_size) > max_dimension:
            new_size = small_size

    return tf.constant(new_size + [num_channels])

def _compute_new_dynamic_size(image, min_dimension, max_dimension):
    """Compute new dynamic shape for resize_to_range method."""
    image_shape = tf.shape(image)
    orig_height = tf.to_float(image_shape[0])
    orig_width = tf.to_float(image_shape[1])
    num_channels = image_shape[2]
    orig_min_dim = tf.minimum(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    min_dimension = tf.constant(min_dimension, dtype=tf.float32)
    large_scale_factor = min_dimension / orig_min_dim
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = tf.to_int32(tf.round(orig_height * large_scale_factor))
    large_width = tf.to_int32(tf.round(orig_width * large_scale_factor))
    large_size = tf.stack([large_height, large_width])
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = tf.maximum(orig_height, orig_width)
        max_dimension = tf.constant(max_dimension, dtype=tf.float32)
        small_scale_factor = max_dimension / orig_max_dim
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = tf.to_int32(tf.round(orig_height * small_scale_factor))
        small_width = tf.to_int32(tf.round(orig_width * small_scale_factor))
        small_size = tf.stack([small_height, small_width])
        new_size = tf.cond(
            tf.to_float(tf.reduce_max(large_size)) > max_dimension,
            lambda: small_size, lambda: large_size)
    else:
        new_size = large_size
    return tf.stack(tf.unstack(new_size) + [num_channels])


def resize_to_range(image,
                    label=None,
                    fixed_input_height=None,
                    fixed_input_width=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=True,
                    pad_to_max_dimension=False):
    if len(image.get_shape()) != 3:
        raise ValueError('Image should be 3D tensor')

    image.set_shape([fixed_input_height, fixed_input_width, 3])
    label.set_shape([fixed_input_height, fixed_input_width, 1])

    def _resize(item):
        if item.get_shape().is_fully_defined():
            new_size = _compute_new_static_size(item,
                min_dimension, max_dimension)
        else:
            new_size = _compute_new_dynamic_size(item,
                min_dimension, max_dimension)
        new_item = tf.image.resize_images(
            item, new_size[:-1], method=method, align_corners=align_corners)

        if pad_to_max_dimension:
            new_item = tf.image.pad_to_bounding_box(
            new_item, 0, 0, max_dimension, max_dimension)
        return new_item

    with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
        if min_dimension and max_dimension:
            new_image = _resize(image)
            new_label = _resize(label)
            result = [new_image, new_label]

    return result


def random_scale(image,
                 label,
                 min_scale_ratio=0.5,
                 max_scale_ratio=2.0,
                 seed=_RANDOM_PREPROCESSOR_SEED,
                 preprocess_vars_cache=None):
    with tf.name_scope('RandomScale', values=[image]):
        result = []
        image_height, image_width = tf.shape(image)
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
        image = tf.image.resize_images(
            image, [image_newysize, image_newxsize], align_corners=True)
        result.append(image)
        if label:
            label = tf.image.resize_nearest_neighbor(
                label, [image_newysize, image_newxsize], align_corners=True)
            result.append(label)
    return tuple(result)


def random_crop(
        image,
        label=None,
        aspect_ratio_range=(0.75, 1.33),
        area_range=(0.1, 1.0),
        preprocess_vars_cache=None):
    with tf.name_scope('RandomCropImage', values=[image, label]):
        result = []
        image_shape = tf.shape(image)
        generator_func = functools.partial(
            tf.image.sample_distorted_bounding_box,
            image_size=image_shape,
            bounding_boxes=[],
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=100)

        sample_distorted_bounding_box = _get_or_create_preprocess_rand_vars(
            generator_func,
            _IMAGE_CROP_KEY,
            preprocess_vars_cache)

        im_box_begin, im_box_size, im_box = sample_distorted_bounding_box

        new_image = tf.slice(image, im_box_begin, im_box_size)
        new_image.set_shape([None, None, image.get_shape()[2]])
        new_image.append(label)

        if label is not None:
            new_label = tf.slice(label, im_box_begin, im_box_size)
            new_label.set_shape([None, None, label.get_shape()[2]])
            result.append(label)
    return tuple(result)


def random_vertical_flip(image,
                         label=None,
                         seed=_RANDOM_PREPROCESSOR_SEED,
                         preprocess_vars_cache=None):

    def _flip_image(image):
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped

    with tf.name_scope('RandomVerticalFlip', values=[image, label]):
        result = []
        # random variable defining whether to do flip or not
        generator_func = functools.partial(tf.random_uniform, [], seed=seed)
        do_a_flip_random = _get_or_create_preprocess_rand_vars(
            generator_func, _IMAGE_VERTICAL_FLIP_KEY,
            preprocess_vars_cache)
        do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

        image = tf.cond(do_a_flip_random,
            lambda: _flip_image(image), lambda: image)
        result.append(image)

        if label is not None:
            image = tf.cond(do_a_flip_random,
                lambda: _flip_image(label), lambda: label)
            result.append(image)

    return tuple(result)


def random_horizontal_flip(image,
                           label=None,
                           seed=_RANDOM_PREPROCESSOR_SEED,
                           preprocess_vars_cache=None):

    def _flip_image(image):
        # flip image
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, label]):
        result = []
        # random variable defining whether to do flip or not
        generator_func = functools.partial(tf.random_uniform, [], seed=seed)
        do_a_flip_random = _get_or_create_preprocess_rand_vars(
            generator_func, _IMAGE_HORIZONTAL_FLIP_KEY,
            preprocess_vars_cache)
        do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

        image = tf.cond(do_a_flip_random,
            lambda: _flip_image(image), lambda: image)
        result.append(image)

        if label is not None:
            image = tf.cond(do_a_flip_random,
                lambda: _flip_image(label), lambda: label)
            result.append(image)

    return tuple(result)


def preprocess_runner(tensor_dict, func_list):

    if dataset_builder._IMAGE_FIELD not in tensor_dict \
      or dataset_builder._LABEL_FIELD not in tensor_dict:
        raise ValueError('"tensor_dict" must have both image'
                         'and label fields')

    for item_key in [dataset_builder._IMAGE_FIELD,
                     dataset_builder._LABEL_FIELD]:
        items = tensor_dict[item_key]
        # if len(items.get_shape()) != 4:
            # raise ValueError('images in tensor_dict should be rank 4')
        # items = tf.squeeze(items, squeeze_dims=[0])
        tensor_dict[item_key] = items

    images = tensor_dict[dataset_builder._IMAGE_FIELD]
    labels = tensor_dict[dataset_builder._LABEL_FIELD]
    for preprocessor_step_func in func_list:
        results = preprocessor_step_func(image=images, label=labels)

    tensor_dict[dataset_builder._IMAGE_FIELD] = results[0]
    tensor_dict[dataset_builder._LABEL_FIELD] = results[1]

    return tensor_dict


def build(preprocessor_config_list):
    proprocessor_func_list = []

    # Note that either aspect_ratio_image_resize or random_crop must be used
    # in order to provide a static shape to the model during construction
    #
    # For aspect_ratio_image_resize, specify fixed_input_height and
    # fixed_input_wdith
    #
    # For random_rotation, specify crop_width, crop_height
    # if preprocessor_config_list.HasField('aspect_ratio_image_resize') and \
    #   not preprocessor_config_list.HasField('crop'):
    #     resize_config = preprocessor_config_list.aspect_ratio_image_resize
    #     if not resize_config.HasField('fixed_input_height') or \
    #       not resize_config.HasField('fixed_input_wdith'):
    #         raise ValueError('Either use "aspect_ratio_image_resize" with '
    #                         '"fixed_input_wdith" and "fixed_input_height" '
    #                         'or use "random_image_crop"')

    for preprocessor_step_config in preprocessor_config_list:
        step_type = preprocessor_step_config.WhichOneof('preprocessing_step')

        # Resize the image and keep the aspect_ratio
        if step_type == 'aspect_ratio_image_resize':
            config = preprocessor_step_config.aspect_ratio_image_resize
            if not (config.min_dimension <= config.max_dimension):
                raise ValueError('min_dimension > max_dimension')
            method = RESIZE_METHOD_MAP[config.resize_method]
            image_resizer_fn = functools.partial(
                resize_to_range,
                fixed_input_width=config.fixed_input_width,
                fixed_input_height=config.fixed_input_height,
                min_dimension=config.min_dimension,
                max_dimension=config.max_dimension,
                pad_to_max_dimension=config.pad_to_max_dimension)
            proprocessor_func_list.append(image_resizer_fn)

        # Randomly Scale the image
        if step_type == 'random_image_scale':
            config = preprocessor_step_config.random_image_scale
            if not (config.max_scale_ratio <= config.min_scale_ratio):
                raise ValueError('min_scale_ratio > max_scale_ratio')

            image_scale_fn = functools.partial(
                random_scale,
                min_scale_ratio=config.min_scale_ratio,
                max_scale_ratio=config.max_scale_ratio)
            proprocessor_func_list.append(image_scale_fn)

        # Random Flips and Rotations
        if step_type == 'random_horizontal_flip':
            config = preprocessor_step_config.random_horizontal_flip
            image_horizontal_flip_fn = functools.partial(
                random_horizontal_flip)
            proprocessor_func_list.append(image_horizontal_flip_fn)

        if step_type == 'random_vertical_flip':
            config = preprocessor_step_config.random_vertical_flip
            image_vertical_flip_fn = functools.partial(
                random_horizontal_flip)
            proprocessor_func_list.append(image_vertical_flip_fn)

        if step_type == 'random_rotation':
            raise ValueError('"random_rotation" preprocessor step'
                             'has not been implemented yet')

        # Randomly crop the image. Be sure to add padding first if needed
        if step_type == 'random_image_crop':
            config = preprocessor_step_config.random_image_crop
            if not (config.min_aspect_ratio <= config.max_aspect_ratio):
                raise ValueError('min_aspect_ratio > max_aspect_ratio')
            if not (config.min_area <= config.max_area):
                raise ValueError('min_area > max_area')

            aspect_ratio_range = (
                config.min_aspect_ratio, config.max_aspect_ratio)
            area_range = (config.min_area, config.max_area)

            image_crop_fn = functools.partial(
                random_crop,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range)
            proprocessor_func_list.append(image_crop_fn)

        if len(proprocessor_func_list) <= 0 and \
            len(preprocessor_config_list) > 0:
            raise ValueError('Unknown preprocessing step.')

    preprocessor = functools.partial(
        preprocess_runner,
        func_list=proprocessor_func_list)

    return preprocessor
