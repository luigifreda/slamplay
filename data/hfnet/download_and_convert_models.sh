#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. $SCRIPT_DIR/../../config.sh  # source configuration file and utils

# You can get your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# Alternatively, you can use the following procedure to download TensorRT 8.5.1.7 from the My Drive.

INSTALL_PATH=${SCRIPT_DIR}

FILE_NAME1="hfnet_tf.tar.xz" 
URL_MY_DRIVE_ID1="13l6Gagk5eXnHXEFgS5gO123Y61G-o0aP" # full file https://drive.google.com/file/d/13l6Gagk5eXnHXEFgS5gO123Y61G-o0aP/view?usp=sharing

FILE_NAME2="hfnet-rt.tar.xz" 
URL_MY_DRIVE_ID2="12gQr4VQkYgrqRCKqHecWR0ZgBxUQqv5a" # full file https://drive.google.com/file/d/12gQr4VQkYgrqRCKqHecWR0ZgBxUQqv5a/view?usp=sharing


# create install path 
if [ ! -d $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi 
cd $INSTALL_PATH


if [ ! -d "${INSTALL_PATH}/hfnet_tf" ]; then
    # download the file 
    if [ ! -f $FILE_NAME1 ]; then
        pip install gdown
        echo Downloading $FILE_NAME1 from drive
        gdrive_download $URL_MY_DRIVE_ID1
    fi
    tar -xvf $FILE_NAME1 
fi 

if [ ! -d "${INSTALL_PATH}/hfnet-rt" ]; then
    # download the file 
    if [ ! -f $FILE_NAME2 ]; then
        pip install gdown
        echo Downloading $FILE_NAME2 from drive
        gdrive_download $URL_MY_DRIVE_ID2
    fi
    tar -xvf $FILE_NAME2 
fi 

cd $SCRIPT_DIR
