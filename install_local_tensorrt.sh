#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

# You can get your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# Alternatively, you can use the following procedure to download TensorRT 8.5.1.7 from my google drive.

USE_GOOGLE_DRIVE=1

INSTALL_PATH=${SCRIPT_DIR}/thirdparty
FILE_NAME=""
if [[ $TENSORRT_VERSION == 10* ]]; then 
  FILE_NAME="TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz" # replace this with your file name if you have one and set USE_GOOGLE_DRIVE=0
  URL_MY_DRIVE_ID="1qgMFeQJYC0U8At87pv4NfBIZaSdAiAXU" # full file https://drive.google.com/file/d/1qgMFeQJYC0U8At87pv4NfBIZaSdAiAXU/view?usp=sharing
elif [[ $TENSORRT_VERSION == 8* ]]; then
  FILE_NAME="TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz" # replace this with your file name if you have one and set USE_GOOGLE_DRIVE=0
  URL_MY_DRIVE_ID="1f1ULBTjhaDIceQGWKACZi5SLzPYAqScr" # full file https://drive.google.com/file/d/1f1ULBTjhaDIceQGWKACZi5SLzPYAqScr/view?usp=drive_link
else 
  echo "TensorRT version $TENSORRT_VERSION is not supported"
  exit -1
fi

# create install path 
if [ ! -d $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi 
cd $INSTALL_PATH


if [ ! -d "${INSTALL_PATH}/TensorRT/lib" ]; then

  if [ $USE_CUDA -eq 1 ]; then
      install_packages libcudnn8 libcudnn8-dev  # check and install otherwise this is going to update to the latest version (and that's not we necessary want to do)
  fi 

  mkdir -p ${INSTALL_PATH}/TensorRT
  pip install gdown
  if [ $USE_GOOGLE_DRIVE -eq 1 ]; then
    # download the file 
    if [ ! -f $FILE_NAME ]; then
      echo Downloading $FILE_NAME from drive
      echo Alternatively, you can manually download your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading and place its path in FILE_NAME variable
      gdrive_download $URL_MY_DRIVE_ID 
    fi
  fi 

  if [ ! -d "${INSTALL_PATH}/TensorRT/bin" ]; then
    tar -xzvf $FILE_NAME -C ${INSTALL_PATH}/TensorRT --strip-components=1
  fi 

fi 


