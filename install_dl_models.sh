#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

print_blue '================================================'
print_blue "Downloading and preparing hfnet models ..."

"$SCRIPT_DIR"/data/hfnet/download_and_convert_models.sh


print_blue '================================================'
print_blue "Downloading and preparing depth-anything models ..."

"$SCRIPT_DIR"/data/depth_anything/download_and_convert_models.py


print_blue '================================================'
print_blue "Downloading and preparing segment-anything models ..."

"$SCRIPT_DIR"/data/segment_anything/download_and_convert_models.py