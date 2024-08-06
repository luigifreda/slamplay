#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils


print_blue '================================================'
print_blue "Downloading new colleget dataset ..."

$SCRIPT_DIR/scripts/download_dataset_new_college.sh


print_blue '================================================'
print_blue "Downloading remode testing dataset ..."

$SCRIPT_DIR/scripts/download_dataset_remode_test_data.sh


print_blue '================================================'
print_blue "Downloading DBoW2 vocabulary ..."

$SCRIPT_DIR/data/dbow2_vocabulary/download_vocabulary.sh