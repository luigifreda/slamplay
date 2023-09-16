#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

MAIN_DIR=$SCRIPT_DIR/..
MAIN_DIR=$(readlink -f $MAIN_DIR)  # this reads the actual path if a symbolic directory is used

DATA_DIR=$MAIN_DIR/data
DATA_DIR=$(readlink -f $DATA_DIR)  # this reads the actual path if a symbolic directory is used

set -euxo pipefail

if [ ! -d $DATA_DIR/remode_test_data ]; then
    cd $DATA_DIR

    wget http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip
    unzip remode_test_data.zip -d remode_test_data
    rm remode_test_data.zip
fi 