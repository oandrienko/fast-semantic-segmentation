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

_IMAGE_VERTICAL_FLIP_KEY = 'IMAGE_VERTICAL_FLIP_STEP'

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
                         labels_channel_dim=1):
    with tf.name_scope('DimensionInput', values=[image, min_dimension]):
        fixed_input_tensor_shape = (
            height_to_set, width_to_set, images_channel_dim)
        images.set_shape(fixed_input_tensor_shape)
        fixed_label_tensor_shape = (
            height_to_set, width_to_set, labels_channel_dim)
        labels.set_shape(fixed_label_tensor_shape)
        return images, labels


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
                 seed=_RANDOM_PREPROCESSOR_SEED,
                 preprocess_vars_cache=None):
    with tf.name_scope('RandomScale', values=[images, labels]):
        image_height, image_width ,_ = tf.shape(images)

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

        # Must be 4D tensor for resize ops
        images = tf.expand_dims(images, 0)
        labels = tf.expand_dims(labels, 0)
        with tf.control_dependencies(shape_assert):
            images = tf.image.resize_bilinear(
                images, [image_newysize, image_newxsize], align_corners=True)
            labels = tf.image.resize_nearest_neighbor(
                labels, [image_newysize, image_newxsize], align_corners=True)
        images = tf.squeeze(images, [0])
        labels = tf.squeeze(labels, [0])

    return images, labels


def random_crop(images, labels,
                crop_height, crop_width,
                images_channel_dim,
                labels_channel_dim,
                preprocess_vars_cache=None):

    def _apply_random_crop(inputs, offsets, crop_shape):

        def my_func1(x):
            print("inputs.shape: ", x.shape)
            return x
        inputs = tf.py_func(my_func1, [inputs], tf.float32)

        def my_func2(x):
            print("offsets: ", x)
            print("offsets.shape: ", x.shape)
            return x
        offsets = tf.py_func(my_func2, [offsets], tf.int32)

        def my_func3(x):
            print("crop_shape: ", x)
            print("crop_shape.shape: ", x.shape)
            return x
        crop_shape = tf.py_func(my_func3, [crop_shape], tf.int32)

        sliced_inputs = tf.slice(inputs, offsets, crop_shape)

        def my_func4(x):
            print("sliced_inputs.shape: ", x.shape)
            return x
        sliced_inputs = tf.py_func(my_func4, [sliced_inputs], tf.float32)

        out_inputs = tf.reshape(sliced_inputs, crop_shape)
        return out_inputs

    with tf.name_scope('RandomCropImage', values=[images, labels]):
        images_shape = tf.shape(images)
        images_height = images_shape[0]
        images_width = images_shape[1]

        # def my_funch(x):
        #     print("image height: ", x)
        #     return x
        # images_height = tf.py_func(my_funch, [images_height], tf.int32)

        # def my_funcw(x):
        #     print("image width: ", x)
        #     return x
        # images_width = tf.py_func(my_funcw, [images_width], tf.int32)

        max_offset_height = tf.reshape(images_height-crop_height+1, [])
        max_offset_width = tf.reshape(images_width-crop_width+1, [])

        # def my_funcc(x):
        #     print("max_offset height: ", x)
        #     return x
        # max_offset_height = tf.py_func(my_funcc, [max_offset_height], tf.int32)

        # def my_funcg(x):
        #     print("max_offset width: ", x)
        #     return x
        # max_offset_width = tf.py_func(my_funcg, [max_offset_width], tf.int32)

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

        # def my_funcll(x):
        #     print("RAND offset height: ", x)
        #     return x
        # offset_height = tf.py_func(my_funcll, [offset_height], tf.int32)

        # def my_funcpp(x):
        #     print("RAND offset width: ", x)
        #     return x
        # offset_width = tf.py_func(my_funcpp, [offset_width], tf.int32)

        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
        crop_shape_images = tf.stack(
            [crop_height, crop_width, images_channel_dim])
        crop_shape_labels = tf.stack(
            [crop_height, crop_width, labels_channel_dim])

        # def my_functttt(x):
        #     print("offsets tensor: ", x)
        #     return x
        # offsets = tf.py_func(my_functttt, [offsets], tf.int32)

        # def my_funckk(x):
        #     print("crop_shape_images tensor: ", x)
        #     return x
        # crop_shape_images = tf.py_func(my_funckk, [crop_shape_images], tf.int32)

        # def my_funcii(x):
        #     print("crop_shape_labels: ", x)
        #     return x
        # crop_shape_labels = tf.py_func(my_funcii, [crop_shape_labels], tf.int32)

        images = _apply_random_crop(images, offsets, crop_shape_images)
        labels = _apply_random_crop(images, offsets, crop_shape_labels)

        # Must set shape here or in the set shape preprocessor step
        # when dealing with ICNet
        if images_channel_dim and labels_channel_dim:
            images.set_shape((crop_height, crop_width, images_channel_dim))
            labels.set_shape((crop_height, crop_width, labels_channel_dim))

        return images, labels


def random_vertical_flip(image,
                         label=None,
                         seed=_RANDOM_PREPROCESSOR_SEED,
                         preprocess_vars_cache=None):

    def _flip_image(image):
        image_flipped = tf.image.flip_up_down(image)
        return image_flipped

    with tf.name_scope('RandomVerticalFlip', values=[image, label]):
        result = []
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
        image_flipped = tf.image.flip_left_right(image)
        return image_flipped

    with tf.name_scope('RandomHorizontalFlip', values=[image, label]):
        result = []
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


def preprocess_runner(tensor_dict, func_list, preprocess_vars_cache=None):
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

    images = tensor_dict[dataset_builder._IMAGE_FIELD]
    labels = tensor_dict[dataset_builder._LABEL_FIELD]
    images_shape = tf.shape(images)
    labels_shape = tf.shape(labels)
    shape_assert = tf.Assert(
        tf.equal(images_shape, labels_shape),
        ["Label and Image shape must match"])
    for preprocessor_step_func in func_list:
        results = preprocessor_step_func(images=images, labels=labels,
                        preprocess_vars_cache=preprocess_vars_cache)

    tensor_dict[dataset_builder._IMAGE_FIELD] = results[0]
    tensor_dict[dataset_builder._LABEL_FIELD] = results[1]
    return tensor_dict


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
            if not (config.max_scale_ratio <= config.min_scale_ratio):
                raise ValueError('min_scale_ratio > max_scale_ratio')

            image_scale_fn = functools.partial(
                random_scale,
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

        if step_type == 'random_vertical_flip':
            config = preprocessor_step_config.random_vertical_flip
            image_vertical_flip_fn = functools.partial(
                random_horizontal_flip)
            proprocessor_func_list.append(image_vertical_flip_fn)

        if len(proprocessor_func_list) <= 0 and \
            len(preprocessor_config_list) > 0:
            raise ValueError('Unknown preprocessing step.')

    preprocessor = functools.partial(
        preprocess_runner,
        func_list=proprocessor_func_list)

    return preprocessor
