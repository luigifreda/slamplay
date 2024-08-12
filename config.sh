#!/usr/bin/env bash

CONFIG_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_DIR=$(readlink -f $CONFIG_DIR)  # this reads the actual path if a symbolic directory is used
cd $CONFIG_DIR # this brings us in the actual folder of this config script (not the symbolic one)
#echo CONFIG_DIR: $CONFIG_DIR


source $CONFIG_DIR/bash_utils.sh

# ====================================================
# Python Settings 
# ====================================================

if [[ $UBUNTU_VERSION == *"24.04"* ]] ; then
    cd $CONFIG_DIR
	if [ ! -d "$CONFIG_DIR/.venv" ]; then
		echo "installing virtualenv under Ubuntu 24.04"
		sudo apt install -y python3-venv
		python3 -m venv .venv
	fi 
	echo "activating python venv $CONFIG_DIR/.venv"
    source $CONFIG_DIR/.venv/bin/activate
    cd -
fi 

# ====================================================
# OpenCV Settings 
# ====================================================

# 1: ON, 0: OFF
export USE_LOCAL_OPENCV=1   # use a local installation of OpenCV

export OPENCV_VERSION="4" # default opencv version  

# or you can set manullay OpenCV_DIR
# export OpenCV_DIR="path to my OpenCV folder"
# export OpenCV_DIR="$CONFIG_DIR/thirdparty/opencv/install/lib/cmake/opencv4"  # here not set 


# ====================================================
# CUDA Settings
# ====================================================

# N.B: if you do not have opencv with CUDA support you must set above:
# USE_LOCAL_OPENCV=1

# 1: ON, 0: OFF
export USE_CUDA=0  # use CUDA 
export CUDA_VERSION="cuda-11.8"  # must be an installed CUDA path in "/usr/local"; 
                                 # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
if [ ! -d /usr/local/$CUDA_VERSION ]; then
    CUDA_VERSION="cuda"  # use last installed CUDA path (standard path)
fi 

export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDADIR=/usr/local/$CUDA_VERSION


# ====================================================
# TensorRT Settings
# ====================================================

export USE_TENSORRT=1  # use TensorRT (will locally install TensorRT and use it. Only available if you installed CUDA and this is properly detected)
#export TENSORRT_VERSION="10" # WIP, does not work yet 
export TENSORRT_VERSION="8"

# ====================================================
# Tensorflow Settings
# ====================================================

# This is working with the configuration <>
export USE_TENSORFLOW=0  # use Tensorflow (will locally install Tensorflow and use it. Only available if you installed tensorflow_cc from source)
export TENSORFLOW_ROOT="$HOME/.tensorflow"
if [ $USE_TENSORFLOW -eq 1 ]; then
	if [ ! -d "$TENSORFLOW_ROOT" ]; then
		echo "TENSORFLOW_ROOT: $TENSORFLOW_ROOT does not exist"
		USE_TENSORFLOW=0
	fi
fi 

# ====================================================
# Check and Manage Settings 
# ====================================================

# auto managed things below ...

# ====================================================
# SIMD

# check SIMD supports 
export HAVE_SSE3=$(gcc -march=native -dM -E - </dev/null | grep SSE3 || :)
export HAVE_SSE4=$(gcc -march=native -dM -E - </dev/null | grep SSE4 || :)
export HAVE_AVX=$(gcc -march=native -dM -E - </dev/null | grep AVX || : )

# ====================================================
# CUDA 

export CUDA_FOUND=0
if [ -f /usr/local/$CUDA_VERSION/bin/nvcc ] || [ -f /usr/bin/nvcc ]; then
	CUDA_FOUND=1
	echo "CUDA found: $CUDA_VERSION"
fi

# reset env var if CUDA lib is not installed 
if [ $CUDA_FOUND -eq 0 ]; then
	USE_CUDA=0
	USE_TENSORRT=0	
	echo 'CUDA env var reset, check your CUDA installation'
	echo 'TensorRT env var reser'
fi

# ====================================================
# OPENCV 

if [[ -n "$OpenCV_DIR" ]]; then
	if [ ! -d $OpenCV_DIR ]; then 
		echo OpenCV_DIR does not exist: $OpenCV_DIR
		exit 1 
	fi 
fi 

# install a local opencv with CUDA support and more
if [ $USE_LOCAL_OPENCV -eq 1 ] && [[ ! -n "$OpenCV_DIR" ]]; then
	. install_local_opencv.sh   # source it in order to run it and get the env var OPENCV_VERSION
	echo OpenCV version: $OPENCV_VERSION
	if [[ $OPENCV_VERSION == 4* ]]; then
		OpenCV_DIR="$CONFIG_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
	else
		OpenCV_DIR="$CONFIG_DIR/thirdparty/opencv/install/share/OpenCV"
	fi
	echo setting OpenCV_DIR: $OpenCV_DIR
    #export LD_LIBRARY_PATH=$CONFIG_DIR/thirdparty/opencv/install/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

# ====================================================
# Tracy Settings
# ====================================================

# Tracy is a great profiler. Details here https://github.com/wolfpld/tracy

export USE_TRACY=1  # use Tracy, you will be able to profile your apps with the profiler
