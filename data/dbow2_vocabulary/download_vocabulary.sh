#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR=$SCRIPT_DIR/../..

. $ROOT_DIR/config.sh  # source configuration file and utils

INSTALL_PATH=${SCRIPT_DIR}

FILE_NAME1="ORBvoc.bin" 
URL_MY_DRIVE_ID1="1uiXEJ1bp6E1sEv4ftEAbGmsxiNioVeoG" # full file https://drive.google.com/file/d/1uiXEJ1bp6E1sEv4ftEAbGmsxiNioVeoG/view?usp=drive_link

FILE_NAME2="ORBvoc.txt" 
URL_MY_DRIVE_ID2="16BxRiJ-ndwCHas8GC0LZ_AjVjfMVIJuc" # full file https://drive.google.com/file/d/16BxRiJ-ndwCHas8GC0LZ_AjVjfMVIJuc/view?usp=drive_link


cd $INSTALL_PATH

if [ ! -f $FILE_NAME1 ]; then
    echo Downloading $FILE_NAME1 from drive
    gdrive_download $URL_MY_DRIVE_ID1
fi

if [ ! -f $FILE_NAME2 ]; then
    echo Downloading $FILE_NAME2 from drive
    gdrive_download $URL_MY_DRIVE_ID2
fi

cd $SCRIPT_DIR
