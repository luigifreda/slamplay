#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. $SCRIPT_DIR/../../config.sh  # source configuration file and utils

if [ ! -d "${SCRIPT_DIR}/weights" ]; then
    mkdir -p ${SCRIPT_DIR}/weights 
fi

if [ ! -d "${SCRIPT_DIR}/models" ]; then
    mkdir -p ${SCRIPT_DIR}/models
fi

FILE_NAME1="sam_vit_l_0b3195.pth" 
URL_MY_DRIVE_ID1="1SNbCD7z1cHwFH0DmA-KE5OmB6zFMcIDn" # full file https://drive.google.com/file/d/1SNbCD7z1cHwFH0DmA-KE5OmB6zFMcIDn/view?usp=drive_link

FILE_NAME2="sam_onnx_example.onnx"
URL_MY_DRIVE_ID2="1VgIuWXycaDcYpeP3UWA1feq62Zon-JIb" # full file https://drive.google.com/file/d/1VgIuWXycaDcYpeP3UWA1feq62Zon-JIb/view?usp=drive_link

FILE_NAME3="vit_l_embedding.onnx"
URL_MY_DRIVE_ID3="1nwKg-CmEj0njHP4aABxW-3PT2ZnewRYF" # full file https://drive.google.com/file/d/1nwKg-CmEj0njHP4aABxW-3PT2ZnewRYF/view?usp=drive_link

# download the file 
cd ${SCRIPT_DIR}/weights
if [ ! -f $FILE_NAME1 ]; then
    pip install gdown
    echo Downloading $FILE_NAME1 from drive
    gdrive_download $URL_MY_DRIVE_ID1
fi
if [ ! -f $FILE_NAME2 ]; then
    pip install gdown
    echo Downloading $FILE_NAME2 from drive
    gdrive_download $URL_MY_DRIVE_ID2
fi
if [ ! -f $FILE_NAME3 ]; then
    pip install gdown
    echo Downloading $FILE_NAME3 from drive
    gdrive_download $URL_MY_DRIVE_ID3
fi

# convert to onnx format if needed

if [[ ! -s "${SCRIPT_DIR}/models/vit_l_embedding.onnx" ]]; then
    cd ${SCRIPT_DIR}
    ./export_sam_model.py --export_embedding_model
fi 


if [[ ! -s "${SCRIPT_DIR}/models/sam_onnx_example.onnx" ]]; then
    cd ${SCRIPT_DIR}
    ./export_sam_model.py --export_sam_model
fi 

# convert to engine format

./onnx_to_tensorrt.sh


