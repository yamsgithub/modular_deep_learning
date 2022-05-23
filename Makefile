# Makefile for neurips xai workshop code
# Type "make" or "make all" to build the complete development environment
# Type "make help" for a list of commands
# Type "make clean" to clean the venv

# Variables for the Makefile
.PHONY = all clean
SHELL := /bin/bash

VIRTUAL_ENV_TARGET := mnn

# Makefile commands, see below for actual builds

## all              : build the venv
all: virtual_env install_torch

## clean	  : remove venv
clean:
	-rm -rf mnn

## help             : show all commands.
# Note the double '##' in the line above: this is what's matched to produce
# the list of commands.
help                : Makefile
	@sed -n 's/^## //p' $<

## virtual_env        : Install/update a virtual environment with needed packages
virtual_env: $(VIRTUAL_ENV_TARGET)

install_torch:
	source mnn/bin/activate; \
	pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html || pip install torch==1.10.2 torchvision==0.11.3;

# Actual Target work here

$(VIRTUAL_ENV_TARGET):
	python -m venv mnn; \
	source mnn/bin/activate; \
	pip install --upgrade pip; \
	pip install setuptools --upgrade; \
        pip install  -r requirements.txt; \
	python -m ipykernel install --user --name mnn --display-name "mnn";	

