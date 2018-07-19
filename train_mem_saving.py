r"""Train with gradient checkpointing

Allows for training with larger batch size than would normally be
permitted. Used to train PSPNet and ICNet to facilitate the required
batch size of 16. Also facilites measuring memory usage. Thanks to Yaroslav
Bulatov and Tim Salimans for their gradient checkpointing implementation.
Their implementation can be found here:
    https://github.com/openai/gradient-checkpointing

For ICNet, the suggested checkpoint nodes are:

    'SharedFeatureExtractor/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu:0'
    'SharedFeatureExtractor/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu:0'
    'SharedFeatureExtractor/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu:0'
    'SharedFeatureExtractor/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu:0'
    'FastPSPModule/Conv/Relu6:0'

Tested on Titan Xp.
"""

# TODO: Do not keep this in master - move to seperate branch
#   called gradient-checkpointing

