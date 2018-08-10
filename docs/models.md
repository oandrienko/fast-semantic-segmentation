# ICNet Model Checkpoints

This project provides ICNet weights in the form of Tensorflow checkpoints trained on Cityscapes for you to get started fine-tuning your own models.

## ICNet modifications
It is important to note that the ICNet model implemented in this project does not exactly match the inference model that was released by the original ICNet author. The primary difference is that the original ICNet uses a unreleased and proprietary version of ResNet50. When investigating their prototxt, you will notice that the input to their ResNet model has 2 additional Conv layers before the first Average Pool. These layers are not found in the original ResNet50 released by Kaiming He. 

To get past this hurdle, I used the open source implementation of ResNet found in TF Slim. Additionally, the ICNet implementation in this project has the second branch stem out from different layer than the original model. However, please note that these modifications are minor and are not major changes.

<p align = 'center'>
<img src = 'imgs/icnet_tensorboard.jpg' width='180x'>
</p>
<p align = 'center'>ICNet model in Tensorboard</p>

## Cityscapes-trained models

The table below contains a download link aswell as information about the model. The listed information is:

* the mean Intersection over Union (mIoU) on the Cityscapes validation set
* the compression factor indicating the percentage of filters removed during pruning
* the number of parameters in the model

| Model name  | Cityscapes mIoU | Compression Factor | MACs | Outputs |
| ------------ | :--------------: | :--------------: | :-------------: | :-------------: |
| [icnet_resnet_v1_cityscapes_train](XXX) | 68.89% | 1.0 | XXM | Semantic Segmentation |

More coming soon...