#!/usr/bin/env bash

. config.sh  # source configuration file and utils 


set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
cd $SCRIPT_DIR # this brings us in the actual used folder (not the possibly symbolic one)


EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi

print_blue '================================================'
print_blue "Configuring and building thirdparty/tracy ..."

cd thirdparty
if [ ! -d "tracy" ]; then
    sudo apt install -y libtbb-dev wayland-scanner++ waylandpp-dev libglfw3-dev libdbus-1-dev
    git clone https://github.com/wolfpld/tracy.git
    cd tracy
    git checkout tags/v0.11.0  
    #git checkout ffb98a972401c246b2348fb5341252e2ba855d00
    git apply ../tracy.patch
    cd ..
fi 
cd tracy
if [ ! -d build ]; then
    mkdir build
fi
if [[ ! -f build/libTracyClient.a ]]; then
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/install" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC"
    make -j 10
    make install
    cd ..
fi
if [[ ! -f $SCRIPT_DIR/thirdparty/tracy/tracy-profiler ]]; then
    cd profiler/
    if [ ! -d build ]; then
        mkdir build
    fi    
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="$SCRIPT_DIR/install" -DLEGACY=ON
    make -j 10
    cp tracy-profiler $SCRIPT_DIR/thirdparty/tracy
fi

cd $SCRIPT_DIR