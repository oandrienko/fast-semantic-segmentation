## Real-Time Semantic Segmentation in [TensorFlow](https://github.com/tensorflow/tensorflow)

Perform pixel-wise semantic segmentation on high-resolution images in real-time with Image Cascade Network (ICNet), the highly optimized version of the state-of-the-art Pyramid Scene Parsing Network (PSPNet). **This project implements ICNet and PSPNet50 in Tensorflow with training support for Cityscapes.**

<p align = 'center'>
<b><i>Download pre-trained ICNet and PSPNet50 models <a href="docs/model_zoo.md">here</a></i></b>
</i>


<p align = 'center'>
<img src = 'docs/imgs/cityscapes_seq.gif' width = '720px'>
</p>

<p align = 'center'><i>
Deploy ICNet and preform inference at over 30fps on NVIDIA Titan Xp.
</i></p>

This implementation is based off of the original ICNet paper proposed by Hengshuang Zhao titled [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545). Some ideas were also taken from their previous PSPNet paper, [Pyramid Scene Parsing Network](https://arxiv.org/abs/1612.01105
). The network compression implemented is based on the paper [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710
).

### Release information

#### October 14, 2018
An ICNet model trained in August, 2018 has been released as a pre-trained model in the Model Zoo. All the models were trained without coarse labels and are evaluated on the validation set.

#### September 22, 2018
The baseline PSPNet50 pre-trained model files have been released publically in the Model Zoo. The accuracy of the model surpases that referenced in the ICNet paper.

#### August 12, 2018
Initial release. Project includes scripts for training ICNet, evaluating ICNet and compressing ICNet from ResNet50 weights. Also includes scripts for training PSPNet and evaluating PSPNet as a baseline.

## Documentation

  * <a href='docs/installation.md'>**Installation**: Setting up the project</a><br>
  * <a href="docs/datasets.md">**Dataset Format**: Creating TFRecord files for training and evaluation</a><br>
  * <a href="docs/configs.md">**Configs and Finetune Training**: Creating your own configuration files for training and evaluation</a><br>
  * <a href="docs/pspnet.md">**PSPNet50**: Walkthrough for Training PSPNet50 baseline</a><br>
  * <a href="docs/icnet.md">**ICNet**: Walkthrough for Training ICNet with compression</a><br>
  * <a href="docs/model_zoo.md">**Model Zoo**: Links to pre-trained checkpoints</a><br>

#### *Model Depot Inference Tutorials*
  * <a href="https://modeldepot.io/oandrienko/pspnet50-for-urban-scene-understanding">**PSPNet50 Inference Notebook**</a><br>
  * <a href="https://modeldepot.io/oandrienko/icnet-for-fast-segmentation">**ICNet Inference Notebook**</a><br>

## Overview

<p align = 'center'>
<img src = 'docs/imgs/icnet_tensorboard.jpg' width='180x'>
</p>

<p align = 'center'>
ICNet model in Tensorboard.
</p>

### Training ICNet from Classification Weights

This project has implemented the ICNet training process, allowing you to train your own model directly from *ResNet50* weights as is done in the original work. Other available implementations simply convert the Caffe model to Tensorflow, only allowing for fine-tuning from weights trained on Cityscapes.

By training ICNet on weights initialized from ImageNet, you have more flexibility in the transfer learning process. Read more about setting up this process can be found <a href='docs/configs.md'>here</a>. For training ICNet, follow the guide <a href='docs/icnet.md'>here</a>.

### ICNet Network Compression

In order to achieve real-time speeds, ICNet uses a form of network compression called filter pruning. This drastically reduces the complexity of the model by removing filters from convolutional layers in the network. This project has also implemented this ICNet compression process directly in Tensorflow.

The compression is working, however which "compression scheme" to use is still somewhat ambiguous when reading the original ICNet paper. This is still a work in progress.

### PSPNet Baseline Implementation

In order to also reproduce the baselines used in the original ICNet paper, you will also find implementations and pre-trained models for PSPNet50. Since ICNet can be thought of as a modified PSPNet, it can be useful for comparison purposes.

Informtion on training or using the baseline PSPNet50 model can be found <a href='docs/pspnet.md'>here</a>.

## Maintainers
* Oles Andrienko, github: [oandrienko](https://github.com/oandrienko)

If you found the project, documentation and the provided pretrained models useful in your work, consider citing it with

```
@misc{fastsemseg2018,
  author={Andrienko, Oles},
  title={Fast Semantic Segmentation},
  howpublished={\url{https://github.com/oandrienko/fast-semantic-segmentation}},
  year={2018}
}
```


## Related Work

This project and some of the documentation was based on the *Tensorflow Object Detection API*. It was the initial inspiration for this project. The `third_party` directory of this project contains files from *OpenAI's Gradient Checkpointing* project by Tim Salimans and Yaroslav Bulatov. The helper modules found in `third_party/model_deploy.py` are from the Tensorflow Slim project. Finally, another open source ICNet implementation which converts the original Caffe network weights to Tensorflow was used as a reference. Find all these projects below:

* [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
* [Saving memory using gradient-checkpointing](https://github.com/openai/gradient-checkpointing)
* [Tensorflow Slim](https://github.com/tensorflow/models/tree/master/research/slim)
* [ICNet converted from Caffe using Caffe-Tensorflow](https://github.com/hellochick/ICNet-tensorflow)

## Thanks

* This project could not have happened without the advice (and GPU access) given by **Professor Steven Waslander** and **Ali Harakeh** from the *Waterloo Autonomous Vehicles Lab* (now the *Toronto Robotics and Artificial Intelligence Lab*).

