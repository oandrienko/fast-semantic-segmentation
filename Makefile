PROJECT := fast-semantic-segmentation
SHELL := /bin/bash

.PHONY: clean clean-protos clean-pyc build build-protos

help:
	@echo "clean - remove all build outputs"
	@echo "clean-protos - remove compiled artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "build - build required outputs"
	@echo "build-protos - build compiled protos"

clean: clean-protos clean-pyc

clean-protos:
	rm -rf protos/*_pb2.py

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

build: build-protos

build-protos:
	protoc protos/*.proto --python_out=.
