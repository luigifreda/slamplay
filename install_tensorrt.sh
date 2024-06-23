#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

# You can get your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# Alternatively, you can use the following procedure to download TensorRT 8.5.1.7 from the My Drive.

INSTALL_PATH=${SCRIPT_DIR}/thirdparty
FILE_NAME="TensorRT-8.5.1.7.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz" # replace this with your file name if you have one and set USE_DRIVE=0 

USE_DRIVE=1
URL_MY_DRIVE_ID="1f1ULBTjhaDIceQGWKACZi5SLzPYAqScr"


function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function gdrive_download () {
  gdown https://drive.google.com/uc?id=$1
}

# create install path 
if [ ! -d $INSTALL_PATH ]; then
  mkdir -p $INSTALL_PATH
fi 
cd $INSTALL_PATH


if [ ! -d "${INSTALL_PATH}/TensorRT" ]; then
  mkdir -p ${INSTALL_PATH}/TensorRT

  if [ $USE_DRIVE -eq 1 ]; then
    # download the file 
    if [ ! -f $FILE_NAME ]; then
      pip install gdown
      echo Downloading $FILE_NAME from drive
      echo Alternatively, you can manually download your preferred TensorRT version package from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading and place its path in FILE_NAME variable
      gdrive_download $URL_MY_DRIVE_ID 
    fi
  fi 

  if [ ! -d "${INSTALL_PATH}/TensorRT/bin" ]; then
    tar -xzvf $FILE_NAME -C ${INSTALL_PATH}/TensorRT --strip-components=1
  fi 

fi 


