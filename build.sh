#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

echo "running build.sh" 

reset 

# ====================================================

./build_thirdparty.sh 

# ====================================================
# check if we have external options
EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
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
fi

echo "external option: $EXTERNAL_OPTION"

# ====================================================

print_blue '================================================'
print_blue "Building framework"
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
