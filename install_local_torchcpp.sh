#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. $SCRIPT_DIR/config.sh  # source configuration file and utils


if [[ ! -d $SCRIPT_DIR/thirdparty ]]; then
    mkdir -p $SCRIPT_DIR/thirdparty
fi


if [ $USE_CUDA_TORCH -eq 0 ]; then
    CUDA_FOUND=0
fi 

export CUDA_VERSION_CODE=""
if [ $CUDA_FOUND -eq 1 ]; then
    #CUDA_VERSION_CODE=118 # for instance, this stands for 11.8 
    CUDA_VERSION_CODE=$(echo "$CUDA_VERSION_NUMBER" | sed 's/\.//g')
    echo "CUDA_VERSION_CODE: $CUDA_VERSION_CODE"
fi 

cd $SCRIPT_DIR/thirdparty

if [[ ! -d libtorch ]]; then

    if [[ ! -f libtorch.zip ]]; then

        if [ $CUDA_FOUND -eq 1 ]; then
            echo "downloading libtorch with CUDA support"
            # from https://pytorch.org/tutorials/advanced/cpp_frontend.html
            #wget https://download.pytorch.org/libtorch/nightly/cu$CUDA_VERSION_CODE/libtorch-shared-with-deps-latest.zip

            # from https://pytorch.org/get-started/locally/
            wget https://download.pytorch.org/libtorch/cu$CUDA_VERSION_CODE/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu$CUDA_VERSION_CODE.zip -O libtorch.zip
        else
            echo "downloading libtorch on CPU (no CUDA support)"
            # from https://pytorch.org/get-started/locally/
            wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcpu.zip -O libtorch.zip                       
        fi 
    fi 

    if [[ ! -d libtorch ]]; then
        unzip libtorch.zip -d .
    fi 
fi 