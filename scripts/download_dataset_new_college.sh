#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

MAIN_DIR=$SCRIPT_DIR/..
MAIN_DIR=$(readlink -f $MAIN_DIR)  # this reads the actual path if a symbolic directory is used

DATA_DIR=$MAIN_DIR/data
DATA_DIR=$(readlink -f $DATA_DIR)  # this reads the actual path if a symbolic directory is used

set -euxo pipefail

# you can also download the dataset from https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/data.html 
BASE_URL=https://github.com/nicolov/simple_slam_loop_closure/releases/download/0.0.1

if [ ! -d $DATA_DIR/new_college ]; then
    mkdir -p $DATA_DIR/new_college
    cd $DATA_DIR/new_college

    wget -O Images.zip "${BASE_URL}/Images.zip"
    unzip Images.zip

    wget -O ImageCollectionCoordinates.txt "${BASE_URL}/ImageCollectionCoordinates.txt"

    wget -O NewCollegeGroundTruth.mat "${BASE_URL}/NewCollegeGroundTruth.mat"
fi 
