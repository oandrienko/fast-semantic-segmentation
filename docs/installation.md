# Installation

## Dependencies

In order to properly run all scripts in this repository, ensure you have all the following dependencies installed in your environment:

* Tensorflow
* Protobuf 3.0.0
* Numpy
* Scipy
* Matplotlib

## Install libraries using pip

The easiest way to get started is to install all the dependencies using pip. Start off with installing Tensorflow

```
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

Then for the rest of the dependencies

```
pip install --user numpy
pip install --user scipy
pip install --user matplotlib
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
