# Configuration files

In order to train and evaluate a model implemented in this project, you must create a configuration file that define these processes. The configs should follow the form general structure

```
model { ... }

train_config { ... }

train_input_reader { ... }

eval_config { ... }

eval_input_reader { ... }
```

The easiest way to get starting with your own configs is to modify the sample config located at `configs/sample_icnet_resnet_v1.config`.


## Fine-tuning

### Fine-tune ICNet from *ResNet-50* ImageNet weights

If you wish to train ICNet from the original *ResNet50* weights as done by the author of ICNet, you must first download the associated TF Slim *ResNet50* model checkpoints and add point to them in your config file.

First download the TF Slim ResNet-50 weights with

```
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -zxf resnet_v1_50_2016_08_28.tar.gz -C /tmp
```

Fine-tuning options are specified in training configs. You must modify two fields within your `train_config`. We first specify the `fine_tune_checkpoint_type` as classification (instead of segmentation) and enter the the location of the checkpoint in the `fine_tune_checkpoint` field. Your config should look like

```
train_config: {
    ...

    fine_tune_checkpoint_type: "classification"
    fine_tune_checkpoint: "/tmp/resnet_v1_50.ckpt" # resnet50 weights

    ...
}
```

### Fine-tune ICNet from Cityscapes weights

If you wish to use the supplied Cityscapes checkpoints and fine-tune on your own data, then you must specify the `fine_tune_checkpoint_type` field as segmentation. Your config should be in the form

```
train_config: {
    ...

    fine_tune_checkpoint_type: "segmentation"
    fine_tune_checkpoint: "/tmp/train/model.ckpt-YYYY" # cityscapes weights

    ...
}
```
