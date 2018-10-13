# PSPNet in Tensorflow

The Pyramid Scene Parsing Network (PSPNet) is the parent network architecture for ICNet. The original PSPNet Caffe models released by Hengshuang Zhao came in two variants: one with a ResNet50 backbone and the other with a ResNet101 backbone. Since the primary use of PSPNet in this project is for baseline comparisons and training initialization for ICNet, we deal with only PSPNet50.

This project allows you to train PSPNet50 from scratch and use it for inference or as a starting point for initializing ICNet.

## Motivation

### Baseline model for comparison
As in the original ICNet paper, we use PSPNet50 as a baseline model for comparisons against our trained ICNet model. In particular, we compare both accuracy, number of parameters FLOPs and inference time.

### Pre-trained checkpoint for ICNet

As discussed in more detail the <a href='installation.md'>ICNet documentation</a>, PSPNet50 is used to initialize ICNet for training. It has been observed that simply initializing from ResNet50 weights does not allow ICNet to converge to the original accuracy reported by the authors. For this reason, we pre-train PSPNet50 for later use with training ICNet.

## Training Considerations

Training PSPNet50 is relatively straight forward but has various requirements in order to ensure successful outcomes. In particular, the original PSPNet paper mentions good performance can be reached with

* **Large batch sizes** for per-batch Batch Normalization statistics
* **Large image crop size** for preprocessing during training

More detail is provided below regarding these two factors. **In summary, an effective (per-GPU) batch size of 8 or greater and a crop-size of 768 by 768 or greater was shown to be successful for training PSPNet.**

#### Batch Size and Batch Normalization
When training using SGD, a batch size hyperparameter must be specified and defines the number of examples that are used before updating your model weights. Each iteration, Batch Normalization layers estimate a mean and variance based on the given batch of examples. With a larger batch size, the per-batch mean and variance should theoretically be closer to the global population mean and variance. With low batch-sizes, the per-batch statistics can be skewed and cause problems during training. Therefore, the authors of PSPNet suggest using a batch size of 16. However, such a large batch size this is not possible when training on a single 12GB GPU due to the size overall size of PSPNet.

Specifically in Tensorflow, this problem is also not solved even with multi-GPU distributed training. For example, consider splitting a total batch size of 16 across 2 GPUs. This will result in an effective per-GPU batch size of 8. Although the total batch size is 16, Batch Norm statistics for each copy of the model (assuming a data parallelism configuration) are still only calculated based on the effective batch size. The original PSPNet authors get around this issue by using a "Distributed Batch Normalization" Layer specific to Caffe and not currently available in Tensorflow. An [open issue](https://github.com/tensorflow/tensorflow/issues/7439) in the Tensorflow repository exists but it has still not been addressed by the Tensorflow authors.

A solution to this issue is to somehow lower the amount of memory required to train with a decent batch size. This project used [Gradient Checkpointing project](https://github.com/openai/gradient-checkpointing ) from OpenAI to solve this issue. In particular, Gradient Checkpointing allows one to fit larger models on to a single GPU by sacrificing in total training time. In this project, the `train.py` script uses regular gradient decent updates and `train_mem_saving.py` script uses gradient checkpointing. Using a 12GB Titan Xp, a **max batch size of 8** has been found to be sufficient for training PSPNet.

#### Image Cropping For Training
Another important consideration during training is the image crop size. This image crop defines the resolution of images passed into the network for training. Cropping is done as a preprocessing step which is defined in the training configuration files.

The authors of PSPNet state that larger cropsizes are critical for success in training PSPNet. Given the GPU limitation mentioned, a **crop size of 768x768** is used for training the PSPNet50 implementation in this project.

## Training PSPNet From ResNet50 Weights Walkthrough

The following instructions will provide a step-by-step guide for training PSPNet50 the Cityscapes dataset. It is assumed you have access to one or two NVIDIA Titan Xp GPUs or other equivalent graphics cards. Having two GPUs will allow you to run one evaluation process and another training process.

To start, make sure you have setup your training and validation set <a href='datasets.md'> TFRecords created</a> . Then download the TF-Slim ResNet50 checkpoint for initializing PSPNet for training

```
# from the project root
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
mkdir -p resnet_v1_50_2016_08_28
tar -zxvf resnet_v1_50_2016_08_28.tar.gz -C  resnet_v1_50_2016_08_28
```

Next, we setup the configuration file. Copy and modify the supplied configuration file located at:

`configs/pspnet_1.0_713_resnet_v1.config`

It will contain the required hyperparameters for training. You should modify the following fields

```
train_config: {
	# YOUR RESNET CHECKPOINT LOCATION
    fine_tune_checkpoint: "resnet_v1_50_2016_08_28/resnet_v1_50.ckpt"
    ...
}

train_input_reader: {
	# YOUR TRAIN SET TF RECORD PATH
    tf_record_input_reader {
        input_path: "/tmp/cityscapes_train.record"
    }
    ...
}

eval_input_reader: {
	# YOUR VAL SET TF RECORD PATH
    tf_record_input_reader {
        input_path: "/tmp/cityscapes_val.record"
    }
    ...
}
```

To successfully train given the mentioned consideration factors, we will train with gradient checkpointing. We will use the `train_mem_saving.py` script which must be passed the name of nodes used for training. The following nodes can be used for training PSPNet50

* `SharedFeatureExtractor/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu`
* `PSPModule/Conv_4/Relu6`

To start a single-GPU training session, make a directory to save checkpoints to. For example, we make a directory `/tmp/pspnet_1.0_713_resnet_v1_TRAIN`. Then start by running

```
# specify checkpoint nodes for mem saving grads
export CHECKPOINT_NODES=SharedFeatureExtractor/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu,SharedFeatureExtractor/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu,SharedFeatureExtractor/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu,SharedFeatureExtractor/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu,PSPModule/Conv_4/Relu6

# export CUDA_VISIBLE_DEVICES env var if you wish to run on a specific GPU.
python train_mem_saving.py \
	--config_path configs/pspnet_1.0_713_resnet_v1_CUSTOM.config \
	--logdir /tmp/pspnet_1.0_713_resnet_v1_TRAIN \
	--checkpoint_nodes $CHECKPOINT_NODES \
	--test_image_summaries \
	--log_memory
```

To see evaluation results during training, create another directory at `/tmp/pspnet_1.0_713_resnet_v1_EVAL` and then run the evaluation script (targeting another GPU) with

```
# set CUDA_VISIBLE_DEVICES to another GPU
python eval.py \
	--config_path configs/pspnet_1.0_713_resnet_v1_CUSTOM.config \
	--train_dir  /tmp/pspnet_1.0_713_resnet_v1_TRAIN \ 
	--eval_dir /tmp/pspnet_1.0_713_resnet_v1_EVAL \
	--verbose # will log mIoU accuracy
```

Finally, the evaluation script will output all training and evaluation updates and results to a TF Event file. You can monitor training by running Tensorboard with 

```
tensorboard --logdir train:/tmp/pspnet_1.0_713_resnet_v1_TRAIN,eval:/tmp/pspnet_1.0_713_resnet_v1_EVAL
```











