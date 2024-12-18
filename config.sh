#!/usr/bin/env bash

CONFIG_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
CONFIG_DIR=$(readlink -f $CONFIG_DIR)  # this reads the actual path if a symbolic directory is used
cd $CONFIG_DIR # this brings us in the actual folder of this config script (not the symbolic one)
#echo CONFIG_DIR: $CONFIG_DIR


source $CONFIG_DIR/bash_utils.sh

# ====================================================
# BUILD_TYPE 
# ====================================================

export BUILD_TYPE=Release            # control the build type of all the projects
export BUILD_WITH_MARCH_NATIVE=ON    # enable/disable building with --march=native in all the projects

if [[ "$UBUNTU_VERSION" == *"24.04"* ]] ; then
	BUILD_WITH_MARCH_NATIVE=OFF  # At present, building with --march=native does not work under Ubuntu 24.04 (probably due to different default building options in the native libpcl)
fi 

# ====================================================
# C++ standard  
# ====================================================

export CPP_STANDARD_VERSION=20   # we need c++17 since nvcc does not support c++20 yet (probably we can try mixing c++ standards and just let nvcc use c++17 ... not sure this is the best choice)

# ====================================================
# Python Settings 
# ====================================================

UBUNTU_VERSION=$(lsb_release -a 2>&1)  # ubuntu version 
if [[ $UBUNTU_VERSION == *"24.04"* ]] ; then
    cd $CONFIG_DIR
	if [ ! -d "$CONFIG_DIR"/.venv ]; then
		echo "installing virtualenv under Ubuntu 24.04"
		sudo apt install -y python3-venv
		python3 -m venv .venv
	fi 
	echo "activating python venv $CONFIG_DIR/.venv"
    source $CONFIG_DIR/.venv/bin/activate
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
export USE_CUDA=0  # Use CUDA in slamplay code 
export CUDA_VERSION_NUMBER=11.8
export CUDA_VERSION="cuda-$CUDA_VERSION_NUMBER"  # Must be an installed CUDA path in "/usr/local"
if [ ! -d /usr/local/$CUDA_VERSION ]; then
    CUDA_VERSION="cuda"  # Use last installed CUDA path (standard path, which is usually a symbolic link to the last installed CUDA version)
fi 

export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDADIR=/usr/local/$CUDA_VERSION


# ====================================================
# TensorRT Settings
# ====================================================

export USE_TENSORRT=1  # Use TensorRT. The scripts will locally install TensorRT and cmake will use it. 
                       # Only available if you installed CUDA and this is properly detected.
export TENSORRT_VERSION=8 # can be adjusted below depending on the found cuda version 
#export TENSORRT_VERSION=10 # WIP

export TENSORRT_DIR=$CONFIG_DIR/thirdparty/TensorRT # Default value. This is the install path used by the script install_local_tensorrt.sh.

# ====================================================
# Torch Settings
# ====================================================

# Will be used by segment anything 
export USE_TORCH=1  # Use Torch. The scripts will locally install Torch and use it. 
                    # Only available if you installed CUDA and this is properly detected)

export USE_CUDA_TORCH=0  # Use Torch with CUDA support. 1: ON, 0: OFF
						 # It seems that Torch with CUDA support is not working properly (for different mixed deps). 
						 # It's very likely we need to build from source. WIP.

export TORCH_DIR=$CONFIG_DIR/thirdparty/libtorch/share/cmake/Torch # Default value. This is the install path used by the script install_local_libtorch.sh.

# ====================================================
# Tensorflow Settings
# ====================================================

# Will be used by HFNet (one of the available HFNet implementations is based on tensorflow C++ API).
# Tested configuration reported in the README file (check the notes therein):
# - **C++**: 17
# - **TENSORFLOW_VERSION**: 2.9.0 
# - **BAZEL_VERSION**: 5.1.1
# - **CUDA**: 11.6 
# - **CUDNN**: 8.6.0.163-1+cuda11.8       
#
export USE_TENSORFLOW=0  # Use Tensorflow C++ API. Only available if you installed tensorflow_cc from source.
                         # You can use the script install_tensorflow_cc.sh, which will locally install Tensorflow.  
						 # NOTE: This procedures will take a while (~2 hours or so depending on your machine). 
						 #       For this reason, it is required that you manually launch the script install_tensorflow_cc.sh.

export TENSORFLOW_ROOT="$HOME/.tensorflow" # Default value. This is the install path used by the script install_tensorflow_cc.sh.
if [ $USE_TENSORFLOW -eq 1 ]; then
	if [ ! -d "$TENSORFLOW_ROOT" ]; then
		echo "TENSORFLOW_ROOT: $TENSORFLOW_ROOT does not exist"
		USE_TENSORFLOW=0
	fi
fi 

# ====================================================
# Tracy Settings
# ====================================================

# Tracy is a great profiler. Details here https://github.com/wolfpld/tracy

export USE_TRACY=1  # Use Tracy. The script will automatically install it. You will be able to profile your apps with this great profiler.

# ====================================================
# Check and manage settings 
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

# check CUDA and adjust things if needed 
export CUDA_FOUND=0
export CUDA_VERSION_NUMBER=0
if [ -f /usr/local/$CUDA_VERSION/bin/nvcc ] || [ -f /usr/bin/nvcc ]; then
	CUDA_FOUND=1
	echo "CUDA folder found in /usr/local: $CUDA_VERSION"
	CUDA_VERSION_NUMBER=$(get_cuda_version)
	CUDA_VERSION_CODE=$(echo "$CUDA_VERSION_NUMBER" | sed 's/\.//g') # for instance, "118" stands for "cuda 11.8"
	echo "CUDA_VERSION_NUMBER: $CUDA_VERSION_NUMBER"
	echo "CUDA_VERSION_CODE: $CUDA_VERSION_NUMBER"	
fi

if [ $CUDA_FOUND -eq 1 ]; then
	if [ $CUDA_VERSION_CODE -ge 120 ]; then
		echo "CUDA VERSION >= 120"
		export TENSORRT_VERSION="10"
		echo "TENSORRT_VERSION (adjusted): $TENSORRT_VERSION"
	fi
fi 

# Reset env var if CUDA lib is not installed 
if [ $CUDA_FOUND -eq 0 ]; then
	USE_CUDA=0
	CUDA_VERSION_NUMBER=0
	USE_TENSORRT=0	
	USE_TORCH=0
	echo 'CUDA env var reset, check your CUDA installation'
	echo 'TensorRT env var reser'
fi

# ====================================================
# OPENCV 

# Check OpenCV directory exists 
if [[ -n "$OpenCV_DIR" ]]; then
	if [ ! -d $OpenCV_DIR ]; then 
		echo OpenCV_DIR does not exist: $OpenCV_DIR
		exit 1 
	fi 
fi 

# Install a local opencv with CUDA support and more
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
