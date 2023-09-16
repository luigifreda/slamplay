#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

THIRDPARTY_DIR=$SCRIPT_DIR/../thirdparty 
THIRDPARTY_DIR=$(readlink -f $THIRDPARTY_DIR)

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRDPARTY_DIR/g2o/install/lib

$THIRDPARTY_DIR/g2o/install/bin/g2o_viewer $@