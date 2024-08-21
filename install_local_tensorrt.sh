#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

# You can get your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# Alternatively, you can use the following procedure to download TensorRT tar.gz from my google drive.

USE_GOOGLE_DRIVE=1

INSTALL_PATH=${SCRIPT_DIR}/thirdparty

# create install path 
if [ ! -d $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi 
cd $INSTALL_PATH


if [ ! -d "${INSTALL_PATH}/TensorRT/lib" ]; then

  FILE_NAME=""
  
  if [[ $TENSORRT_VERSION == 10* ]]; then 
    if [ $CUDA_VERSION_CODE -le 118 ]; then
      FILE_NAME="TensorRT-10.0.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz" # replace this with your file name if you have one and set USE_GOOGLE_DRIVE=0
      URL_MY_DRIVE_ID="1qgMFeQJYC0U8At87pv4NfBIZaSdAiAXU" # full file https://drive.google.com/file/d/1qgMFeQJYC0U8At87pv4NfBIZaSdAiAXU/view?usp=sharing
    else
      if [ $CUDA_VERSION_CODE -ge 120 ]; then
        FILE_NAME="TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz" # replace this with your file name if you have one and set USE_GOOGLE_DRIVE=0
        URL_MY_DRIVE_ID="1TjyfRAumKmjK2hxDTmKlCsoVgR8tovup" # full filehttps://drive.google.com/file/d/1TjyfRAumKmjK2hxDTmKlCsoVgR8tovup/view?usp=sharing
      else
        echo "TensorRT version $TENSORRT_VERSION is not supported with your cuda $CUDA_VERSION"
        echo "Please, download your TensorRT version (10 or 8), install it into $INSTALL_PATH/TensorRT and set USE_GOOGLE_DRIVE=0" 
        exit -1
      fi 
    fi 
  elif [[ $TENSORRT_VERSION == 8* ]]; then
    FILE_NAME="TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz" # replace this with your file name if you have one and set USE_GOOGLE_DRIVE=0
    URL_MY_DRIVE_ID="1f1ULBTjhaDIceQGWKACZi5SLzPYAqScr" # full file https://drive.google.com/file/d/1f1ULBTjhaDIceQGWKACZi5SLzPYAqScr/view?usp=drive_link
  else 
    echo "TensorRT version $TENSORRT_VERSION is not supported"
    exit -1
  fi

  if [ $CUDA_FOUND -eq 1 ]; then
    if [[ $version == *"24.04"* ]] ; then
        install_packages libcudnn-dev
    else 
        install_packages libcudnn8 libcudnn8-dev  # check and install otherwise this is going to update to the latest version (and that's not we necessary want to do)
    fi
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


