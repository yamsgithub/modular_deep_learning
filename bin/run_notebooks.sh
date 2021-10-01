#!/bin/bash


SCRIPT_PATH="${BASH_SOURCE[0]}";
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

export MNN_HOME=$(python -c "import os, sys; sys.stdout.write(os.path.abspath('$SCRIPT_DIR/..')+'\n')")
echo "MNN_HOME  : $MNN_HOME"

export PYTHONPATH=$MNN_HOME
echo "PYTHONPATH: $PYTHONPATH"

deactivate
source mnn/bin/activate
jupyter lab
