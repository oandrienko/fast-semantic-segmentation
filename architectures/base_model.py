"""Abstract semantic segmentation model.

Defines a base class to be used by groups of semantic segmentation models in
the project. Any supporting scripts such as trainers, evaluators and
exporters should only call the methods defined in the abstract class.

The general order of using the models implementing this abstract class can be
as followed for both training and evaluation

Training flow:
inputs -> preprocess -> predict -> loss -> outputs
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf

slim = tf.contrib.slim


class FastSegmentationModel(object):
    """Abstract base class for semantic segmentation models."""

    __metaclass__ = ABCMeta

    def __init__(self, num_classes):
        self._num_classes = num_classes
        self._groundtruth_labels = {}

    @property
    def shared_feature_extractor_scope(self):
        return 'SharedFeatureExtractor'

    @property
    def num_classes(self):
        return self._num_classes

    @abstractmethod
    def preprocess(self, inputs):
        """Proprocessing call for input images.

        This method should be used for any preprocessing to be done before
        running a prediction. Primarily would be used for image resizing
        and zero centering if a model requires it.

        Args:
          inputs: a [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: a [batch, height_out, width_out, channels]
            float32 tensor representing a batch of images.
        """
        pass

    @abstractmethod
    def predict(self, preprocessed_inputs, true_image_shapes):
        """Run model inference on a set of input images.

        When training, the output should be passed to the loss method

        Args:
          preprocessed_inputs: a [batch, height, width, channels] float32
            tensor representing a batch of images.

        Returns:
          prediction_dict: a dictionary holding prediction tensors
        """
        pass

    @abstractmethod
    def loss(self, prediction_dict, true_image_shapes):
        """Using the groundtruth, computes a loss tensor to be used for training.

        Note that the class must have been supplied the groundtruth tensors
        first with the provide groundtruth method.

        Args:
          prediction_dict: a dictionary holding predicted tensors.

        Returns:
          a dictionary mapping loss names to scalar tensors representing
            loss values.
        """
        pass

    def provide_groundtruth(self,
                          groundtruth_labels):
        """Provide groundtruth tensors.

        Args:
          groundtruth_masks: a list of 3-D tf.float32 tensors of
            shape [num_classes, height_in, width_in] containing a overall
            segmentation mask.  If None, no masks are provided.
        """
        self._groundtruth_labels = groundtruth_labels
