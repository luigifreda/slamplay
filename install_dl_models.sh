#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

print_blue '================================================'
print_blue "Downloading hfnet models ..."

"$SCRIPT_DIR"/data/hfnet/install_models.sh


print_blue '================================================'
print_blue "Downloading depth-anything models ..."

python "$SCRIPT_DIR"/data/depth_anything/install_models.py

