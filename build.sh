#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

echo "running build.sh" 

reset 

# ====================================================

./build_thirdparty.sh 

#./install_dl_models.sh

# ====================================================
# check if we have external options
EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi

# check if we set a BUILD_TYPE
if [[ -n "$BUILD_TYPE" ]]; then
    echo "BUILD_TYPE: $BUILD_TYPE" 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_BUILD_TYPE=$BUILD_TYPE"
else
    echo "setting BUILD_TYPE to Release by default"
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_BUILD_TYPE=Release"     
fi

# check if we set BUILD_WITH_MARCH_NATIVE
if [[ -n "$BUILD_WITH_MARCH_NATIVE" ]]; then
    echo "BUILD_WITH_MARCH_NATIVE: $BUILD_WITH_MARCH_NATIVE" 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DBUILD_WITH_MARCH_NATIVE=$BUILD_WITH_MARCH_NATIVE"
fi

# check if we set a C++ standard
if [[ -n "$CPP_STANDARD_VERSION" ]]; then
    echo "CPP_STANDARD_VERSION: $CPP_STANDARD_VERSION" 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCPP_STANDARD_VERSION=$CPP_STANDARD_VERSION"
fi

# check the use of local opencv
if [[ -n "$OpenCV_DIR" ]]; then
    echo "OpenCV_DIR: $OpenCV_DIR" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DOpenCV_DIR=$OpenCV_DIR"
fi

if [[ $OPENCV_VERSION == 4* ]]; then
    #echo " "	
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DOPENCV_VERSION=4"
fi

# check CUDA options
if [ $USE_CUDA -eq 1 ]; then
    echo "USE_CUDA: $USE_CUDA" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_CUDA=ON"
    #EXTERNAL_OPTION="$EXTERNAL_OPTION -DCUDA_DIR=$CUDADIR"
fi

# check TENSORRT options
if [ $USE_TENSORRT -eq 1 ]; then
    echo "USE_TENSORRT: $USE_TENSORRT" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_TENSORRT=ON -DTensorRT_DIR=$SCRIPT_DIR/thirdparty/TensorRT"
fi

# check TENSORFLOW options
if [ $USE_TENSORFLOW -eq 1 ]; then
    echo "USE_TENSORFLOW: $USE_TENSORFLOW" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_TENSORFLOW=ON -DTENSORFLOW_ROOT=$TENSORFLOW_ROOT"
fi

# check TORCH options
if [ $USE_TORCH -eq 1 ]; then
    echo "USE_TORCH: $USE_TORCH" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_TORCH=ON -DTorch_DIR=$TORCH_DIR"
fi

# check Tracy option 
if [ $USE_TRACY -eq 1 ]; then
    echo "USE_TRACY: $USE_TRACY" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_TRACY_PROFILER=ON"
fi

echo "external option: $EXTERNAL_OPTION"

# ====================================================

print_blue '================================================'
print_blue "Building slamplay framework"
print_blue '================================================'

if [ ! -d build ]; then 
    mkdir build 
    cd build 
    cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION
    cd .. 
fi 

cd $SCRIPT_DIR/build
cmake .. -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION
make -j 8
