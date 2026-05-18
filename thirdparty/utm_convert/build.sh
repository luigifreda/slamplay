#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

cd $SCRIPT_DIR_

ROOT_DIR=$(readlink -f $SCRIPT_DIR_/..)

#. $ROOT_DIR/config.sh  # source configuration file and utils

OPTIONS=$1
RELEASE_TYPE=Release

if [ ! -d build ]; then
    mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=$RELEASE_TYPE $OPTIONS
make -j 8
make install
cd ..