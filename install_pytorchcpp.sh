#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

#. config.sh  # source configuration file and utils


CUDA_VERSION_CODE=118 # this stands for 11.8 

if [[ ! -d thirdparty ]]; then
    mkdir thirdparty
fi

cd thirdparty

if [[ ! -d libtorch ]]; then

    if [[ ! -f libtorch.zip ]]; then
        # from https://pytorch.org/tutorials/advanced/cpp_frontend.html
        #wget https://download.pytorch.org/libtorch/nightly/cu$CUDA_VERSION_CODE/libtorch-shared-with-deps-latest.zip

        # from https://pytorch.org/get-started/locally/
        wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu$CUDA_VERSION_CODE.zip - o libtorch.zip
    fi 

    if [[ ! -d libtorch ]]; then
        unzip libtorch.zip -d .
    fi 
fi 