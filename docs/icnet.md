# Training ICNet with Tensorflow

## Motivation

### Real-time Semantic Segmentation

Due to the various architectural optimizations, ICNet is one of the few proposed models that can perform semantic segmentation in real-time. Other models such as SegNet are similar in efficiency, but are far less accurate.

## Implementation Details
It is important to note that the ICNet model implemented in this project does not exactly match the inference model that was released by the original ICNet author. The primary difference is that the original ICNet uses a unreleased and proprietary version of ResNet50. When investigating their prototxt, you will notice that the input to their ResNet model has 2 additional Conv layers before the first Average Pool. These layers are not found in the original ResNet50 released by Kaiming He.

To get past this hurdle, an open source implementation of ResNet found in TF Slim was used. Additionally, the ICNet implementation in this project has the second branch stem out from a different layer than the original model. However, these modifications are minor and should not change the performance.

## Training ICNet with Tensorflow

### Single-Stage Training Procedure
If you wish to avoid going through the trouble of training multiple times, you can train ICNet directly from classifications weights. This will be similar to what is done in the PSPNet walkthrough <a href="pspnet.md">here</a>. The same general procedure can be followed with ICNet. Note that this will produce worse results than the two-stage procedure.

### Two-Stage Training Procedure
In order to replicate the training procedure in the original ICNet paper, multiple steps must be taken. In particular, transfer learning must be done from the baseline PSPNet50 model. Compression must also then be used. 

1. **Stage 0 ~ Pre-train a PSPNet50 model:** First, a PSPNet50 model is trained on weights initialized from a dilated ResNet50 model. Using a similar training procedure as described in the original paper (with a crop size of 768, 120K training iterations and an initial learning rate of 0.01), the PSPNet50 model in this project was trained and converged at approximately *74% mIoU*.
2. **Stage 1 ~ Initialize ICNet Branch from PSPNet50:** With a base PSPNet50 model trained, the second stage of training can begin by initializing the ICNet quarter resolution branch with the pre-trained PSPNet50 model (with a crop size of 1024, 200K training iterations and an initial learning rate of 0.001). Initializing ICNet from these weights allowed for convergence at accuracies similar to the original ICNet paper.
3. **Stage 2 ~ Compression and Retraining:** Once the base ICNet model is trained, we must prune half of the kernels to achieve the performance of the original paper. This is a process where kernels are removed from each convolutional layer iteratively. After the kernels are pruned, the pruned model must be retrained a final time to recover from the lost accuracy during pruning.

## Training ICNet with Two Stages Walkthrough

The following instructions will provide a step-by-step guide for training ICNet on the Cityscapes dataset. It is assumed you have access to one or two NVIDIA Titan 1080 Ti GPUs or other equivalent graphics cards.

To start, it is assumed the PSPNet walkthrough was followed (as detailed  <a href="pspnet.md">here</a>) to obtain TFRecords for the training set and validation set. You should also have either trained your own PSPNet50 model or downloaded a pre-trained version of the model from the Model Zoo <a href="model_zoo.md">here</a>.

### Stage 1 - PSPNet Finetuning

We start with the first stage by training ICNet from PSPNet50 weights. Assuming you will be using the pretrained model, first download and extract the model

```
# Download the archive from the Google Drive link then
mkdir -p 0818_pspnet50_1.0_713_resnet_v1
tar -zxvf 0818_pspnet50_1.0_713_resnet_v1.tar.gz -C 0818_pspnet50_1.0_713_resnet_v1
```

Next, we setup the configuration file. Copy and modify the supplied stage 1 configuration file located at:

`configs/two_stage_icnet_1.0_953_resnet_v1_stage_1.config`

It will contain the required hyperparameters for training. You must specify your dataset location as usual. You should modify the following checkpoint field with the location of your PSPNet checkpoint file

```
train_config: {
	# YOUR PSPNet CHECKPOINT LOCATION
    fine_tune_checkpoint: "0818_pspnet50_1.0_713_resnet_v1/model.ckpt"
    ...
}
```

Using this config file, we can start the first stage training process. As with PSPNet, memory will most likely be limited so we will train with gradient checkpointing. We will use the `train_mem_saving.py` script. The following nodes can be used for training ICNet with gradient checkpointing

* `SharedFeatureExtractor/resnet_v1_50/block1/unit_3/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block2/unit_4/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block3/unit_6/bottleneck_v1/Relu`
* `SharedFeatureExtractor/resnet_v1_50/block4/unit_3/bottleneck_v1/Relu`
* `FastPSPModule/Conv/Relu6`
* `CascadeFeatureFusion_0/Relu`
* `CascadeFeatureFusion_1/Relu`

To start a single-GPU training session, make a directory to save checkpoints to. For example, we make a directory `/tmp/icnet_1.0_953_resnet_v1_stage_1_TRAIN`. Then start training on the first GPU by running

```
# The checkpointing nodes are supplied in the script by default for ICNet
python train_mem_saving.py \
    --config_path configs/two_stage_icnet_1.0_953_resnet_v1_stage_1.config_CUSTOM.config \
    --logdir /tmp/icnet_1.0_953_resnet_v1_stage_1_TRAIN \
    --test_image_summaries \
    --log_memory
```

To see evaluation results during training, create another directory at `/tmp/icnet_1.0_953_resnet_v1_stage_1_EVAL` and then run the evaluation script (targeting another GPU) with

```
# set CUDA_VISIBLE_DEVICES to another GPU
python eval.py \
    --config_path configs/two_stage_icnet_1.0_953_resnet_v1_stage_1.config_CUSTOM.config \
    --train_dir /tmp/icnet_1.0_953_resnet_v1_stage_1_TRAIN \
    --eval_dir /tmp/icnet_1.0_953_resnet_v1_stage_1_EVAL \
    --verbose # will log mIoU accuracy
```

Once training has finished, use Tensorboard to find the checkpoint with the highest resulting mIoU. We will use this for training the second stage. To open Tensorboard run

```
tensorboard --logdir /tmp/icnet_1.0_953_resnet_v1_stage_1_EVAL
```

### Stage 2 - Compression and Retraining

We will create a directory for storing our compressed model at `/tmp/icnet_1.0_953_resnet_v1_stage_1_COMPRESS`. Run the compression script with

```
python compress.py \
    --prune_config configs/compression/icnet_resnet_v1_pruner_v2.config \
    --input_checkpoint /tmp/icnet_1.0_953_resnet_v1_stage_1_TRAIN/model.ckpt-116511 \
    --output_dir /tmp/icnet_1.0_953_resnet_v1_stage_1_COMPRESS \
    --compression_factor 0.5 \  # We will compress to half
    --interactive               # If we want to visualize the kernels being removed
```

Now that we have a compressed model, we need to retrain. As with the first stage config, copy and modify the supplied stage 2 configuration file located at:

`configs/two_stage_icnet_1.0_953_resnet_v1_stage_2.config`

It will contain the required hyperparameters for training. You should modify same fields with before, making sure to point to the compressed model for initialization. Notice that in this config, the `filter_scale` field is set to 0.5 instead of 1.0. Then run training and evaluation as before which will produce your final model.
