# Installation

## Dependencies

In order to properly run all scripts in this repository, ensure you have all the following dependencies installed in your environment:

* **Tensorflow** (tested with v1.8)
* **Protobuf 3.0.0**
* **Numpy**
* Matplotlib (for compression visualization)
* PIL (for loading images for inference)
* Toposort (for memory saving gradients support)

## Install libraries using pip

The easiest way to get started is to install all the dependencies using pip. Start off with installing Tensorflow

```
# For CPU
pip install tensorflow==1.8.0
# For GPU
pip install tensorflow-gpu==1.8.0
```

Then for the rest of the main dependencies

```
pip install --user numpy matplotlib pillow toposort
```

## Install Protobuf Compiler

To install protoc, the protobuf compiler on Ubuntu, run

```
sudo apt-get install protobuf-compiler
```

If you are on a Mac, then use brew and run

```
brew install protobuf
```

# Compiling Proto files
In to run most of the scripts in this projects, make sure to compile the proto files with your installed version of protoc. You can run the following from the project root

```
make build-protos
```

or explicitly with

```
protoc protos/*.proto --python_out=.
```
