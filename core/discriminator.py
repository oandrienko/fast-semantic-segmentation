""" Domain Discriminator for Unsupervised Domain Adaptation.

Defines a class to be used by the trainer when doing adveserial
domain adaptation. Implemented as a simple classification
model which takes in logits form a given feature extractor
and predicts whether the input is from a source domain
or target domain.

Useful tips: For the discriminator to work we must first train
the feature extracter on the source data. After this is done,
the segmentation output is replaced with this domain classifier
in order to fine tune the predictions.

See https://arxiv.org/pdf/1505.07818.pdf for details
"""

class DomainDiscriminatorClassifier(object):
    """ Domain classifier for Adveserial training."""

    def __init__(self, adveserial_weight):
        self._num_classes = 2
        self._groundtruth_lists = {}
        raise ValueError('Unimplemented...')

    def predict(self, feature_extracter_features):
        pass

    def generate_groundtruth(self, labels):
        pass

    def loss(self, predictions):
        pass
