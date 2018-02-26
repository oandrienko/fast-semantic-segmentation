PROJECT := fast-semantic-segmentation
SHELL := /bin/bash

export-path:
	export PYTHONPATH=$PYTHONPATH
build-protos:
	protoc protos/*.proto --python_out=.
clean-protos:
	rm -rf protos/*_pb2.py
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
