# Pre-trained ICNet and PSPNet Models

This project provided pre-trained model weights for both ICNet and  the PSPNet50 baseline model. Both Tensorflow checkpoints and frozen graphs are provided for each model variant, allowing you to either deploy or fine-tune on your own dataset.

Each downloadable archive will contain the following:

* a graph proto (`graph.pbtxt`)
* a checkpoint
  (`model.ckpt.data-00000-of-00001`, `model.ckpt.index`, `model.ckpt.meta`)
* a frozen graph proto (`frozen_inference_graph.pb`) with folded batch norm parameters and weights included in the graph as constants. To be used for out of the box inference
* a config file (`pipeline.config`) which was used to generate the graph.
* a text file with per-layer benchmarking information (`benchmark.txt`)


## Cityscapes-trained models

The table below contains a download link as well as information about the model.The listed information is:

* mean Intersection over Union (mIoU) on the Cityscapes validation set ‡
* compression factor indicating the percentage of filters removed during pruning
* number of parameters in the model
* estimate of floating point operations per second

| Model  | Val. Set mIoU | Compress. Factor | Parameters | FLOPs |
| ------------ | :--------------: | :-------------: | :-------------: | :-------------:
| [0818_pspnet50_1.0_713_resnet_v1](https://drive.google.com/file/d/1ARJOa2QrbSVzQKrSCKaiRVrXUZ52-HXI/view?usp=sharing) | 73.8% | 1.0 | 46.53M | 2942.59B |
| [0818_icnet_0.5_1025_resnet_v1](https://drive.google.com/file/d/1AjeshDkEAAGsgbX9bwLkN6aL9bkd2pgh/view?usp=sharing) | 65.9% | 0.5 | 6.71M | 63.20B |

**‡** **The mIoU here is based on whole-image inference**. This means that evaluation is done with one forward
pass of the network. In the case of Cityscapes, that means the 1024x2048 resolution images are evaluated without cropping and without evaluating subdivided patches (sliding evaluation).
___

### Validating Pre-trained Models

Listed here is an example of checking the accuracy and inference results of a model downloaded from the above list. We will use the PSPNet50 as an example. Start by downlaoding and then extracting the model files with

```
# Download using Google Drive link provided then extract
tar -zxvf 0818_pspnet_1.0_713_resnet_v1.tar.gz
```

You will have multiple formats of the model which are listed above. Edit the configuration file `pipeline.config` with the paths to your Cityscapes train and val TFRecords. Then, to evaluate the checkpoint run

```
python eval.py \
	--train_dir 0818_pspnet_1.0_713_resnet_v1 \
	--eval_dir 0818_pspnet_1.0_713_resnet_v1 \
	--config_path 0818_pspnet_1.0_713_resnet_v1/pipeline.config \
	--evaluate_all_from_checkpoint model.ckpt \
	--verbose
```

The resulting accuracy measure in <mIoU> will be almost identical if you were to run the official Cityscapes per-pixel evaluation script.

To use the model out the box for inference, we must specify either a directory with images or a single image path. We use the inference script like so for Cityscapes size images

```
python inference.py \
	--input_shape 1024,2048,3 \
	--pad_to_shape 1025,2049 \
	--input_path /tmp/DIR_WITH_IMAGES_OR_IMAGE_PATH \
	--config_path 0818_pspnet_1.0_713_resnet_v1/pipeline.config \
	 --trained_checkpoint 0818_pspnet_1.0_713_resnet_v1/model.ckpt \
	--output_dir /tmp/RESULTS_DIR \
	--label_ids # set this if you need labelID outputs, else will be colour

```
