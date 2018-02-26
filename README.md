## Real-Time Semantic Segmentation in [TensorFlow](https://github.com/tensorflow/tensorflow)

Perform pixel-wise semantic segmentation on high-resolution images in real-time.

<p align = 'center'>
<img src = 'docs/imgs/waterloo_test.jpg' height = '246px'>
<img src = 'docs/imgs/cityscapes_val.jpg' height = '246px'>
<a href = 'docs/imgs/cityscapes_seq_test.jpg'>
<img src = 'docs/imgs/cityscapes_seq_test.jpg' width = '627px'>
</a>
</p>
<p align = 'center'>
Preform high accuracy semantic segmentation at up to 50fps on a Titan Xp.
</p>



[ADD SOME HIGHLIGHTS ABOUT MODEL ACCURACY HERE ...]




## Installation

### Prerequisites
- Linux or OSX
- Nvidia GPU and CuDNN

### Dependencies
This project has the following dependancies:
- Tensorflow 1.4
- TensorFlow Models (for use of tf Slim)
- Protobuf 2.6
- Pillow 1.0
- Matplotlib

The easiest way to include the Tf-Slim Image Classification library is to
clone the tensorflow models repository and install slim as a python module
```
git clone https://github.com/tensorflow/models
cd models/research/slim
sudo python setup.py build -b tmp
sudo python setup.py install
```

### Getting Started
After having installed the dependancies with the TF-Slim module in our path,
we clone the project with
```
git clone https://github.com/oandrienko/fast-semantic-segmentation
cd fast-semantic-segmentation
```

Next, we compile the configuration protobuf files with
```
make build-protos
```

## Training

## Evaluating

## Exporting Models
