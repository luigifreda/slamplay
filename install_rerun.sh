#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

print_blue '================================================'
print_blue "Configuring and building thirdparty/rerun ..."

cd thirdparty
if [ ! -d rerun ]; then
	sudo apt-get install -y cargo 
    git clone https://github.com/rerun-io/rerun.git rerun
    #git fetch --all --tags # to fetch tags 
    cd rerun
    git checkout 0.14.1
    cd .. 
fi
cd rerun
make_buid_dir
if [[ ! -d install ]]; then
	cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" $EXTERNAL_OPTIONS
	make -j 8
    make install 
fi 
cd $SCRIPT_DIR