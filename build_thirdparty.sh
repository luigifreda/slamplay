#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils 

# ====================================================

print_blue '================================================'
print_blue "Building thirdparty"
print_blue '================================================'

version=$(lsb_release -a 2>&1)

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

# check TENSORRT options
if [ $USE_TENSORRT -eq 1 ]; then
    echo "USE_TENSORRT: $USE_TENSORRT" 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_TENSORRT=ON -DTensorRT_DIR=$SCRIPT_DIR/thirdparty/TensorRT"
fi

echo "external option: $EXTERNAL_OPTION"
# ====================================================

# install ubuntu dependancies!
#./install_dependencies.sh

if [ $USE_TENSORRT -eq 1 ]; then
    print_blue '================================================'
    print_blue "Configuring and installing thirdparty/TensorRT ..."

    ./install_tensorrt.sh
fi


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/pangolin ..."

cd thirdparty
if [ ! -d pangolin ]; then
	sudo apt-get install -y libglew-dev
	git clone --recursive https://github.com/stevenlovegrove/Pangolin.git pangolin
    #git fetch --all --tags # to fetch tags 
    cd pangolin
    #git checkout tags/v0.6
    git checkout fe57db532ba2a48319f09a4f2106cc5625ee74a9
    git apply ../pangolin.patch  # applied to commit fe57db532ba2a48319f09a4f2106cc5625ee74a9
    cd .. 
fi
cd pangolin
make_buid_dir
if [[ ! -f build/src/libpangolin.so && ! -f build/libpango_core.so ]]; then
	cd build
    PANGOLIN_OPTIONS="-DBUILD_EXAMPLES=OFF"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $PANGOLIN_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/sophus ..."

cd thirdparty
if [ ! -d sophus ]; then
    sudo apt-get install gfortran libc++-dev libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libceres-dev 
    sudo apt install libfmt-dev
	git clone https://github.com/strasdat/Sophus.git sophus
    #git fetch --all --tags # to fetch tags 
    cd sophus
    git checkout 61f9a9815f7f5d4d9dcb7f4ad9f4f42ab3563108
    git apply ../sophus.patch     
    cd .. 
fi
# build is not required 
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/ceres ..."

cd thirdparty
if [ ! -d ceres ]; then
	git clone https://ceres-solver.googlesource.com/ceres-solver ceres
    #git fetch --all --tags # to fetch tags 
    cd ceres
    git checkout tags/2.1.0  # f68321e7de8929fbcdb95dd42877531e64f72f66  
    cd .. 
fi
cd ceres
make_buid_dir
if [[ ! -f install/lib/libceres.a ]]; then
	cd build
    CERES_OPTIONS="-DBUILD_EXAMPLES=OFF"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $CERES_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/gtsam ..."

cd thirdparty
if [ ! -d gtsam ]; then
	git clone https://github.com/borglab/gtsam.git gtsam
    #git fetch --all --tags # to fetch tags 
    cd gtsam
    git checkout tags/4.2a9   
    cd .. 
fi
cd gtsam
make_buid_dir
if [[ ! -f install/lib/libgtsam.so ]]; then
	cd build
    # NOTE: gtsam has some issues when compiling with march=native option!
    # https://groups.google.com/g/gtsam-users/c/jdySXchYVQg
    # https://bitbucket.org/gtborg/gtsam/issues/414/compiling-with-march-native-results-in 
    GTSAM_OPTIONS="-DGTSAM_USE_SYSTEM_EIGEN=On -DGTSAM_BUILD_WITH_MARCH_NATIVE=Off"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $GTSAM_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi
cd $SCRIPT_DIR

# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/g2o ..."

cd thirdparty
if [ ! -d g2o ]; then
    sudo apt install -y libqglviewer-dev-qt5  # to build g2o_viewer 
	git clone https://github.com/RainerKuemmerle/g2o.git g2o
    #git fetch --all --tags # to fetch tags 
    cd g2o
    git checkout tags/20230223_git   
    cd .. 
fi
cd g2o
make_buid_dir
if [[ ! -f install/lib/libg2o_core.so ]]; then
	cd build
    G2O_OPTIONS="-DBUILD_WITH_MARCH_NATIVE=ON -DG2O_BUILD_EXAMPLES=OFF"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $G2O_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi 
cd $SCRIPT_DIR

# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/mahi-gui ..."

cd thirdparty
if [ ! -d mahigui ]; then
    git clone https://github.com/mahilab/mahi-gui.git mahigui
    #git fetch --all --tags # to fetch tags 
    cd mahigui
    git checkout c34f1e22041d5cb88edf76cce6cfe3f44f062581    
    git apply ../mahigui.patch 
    cd .. 
fi
cd mahigui
make_buid_dir
if [[ ! -f install/lib/libmahi-gui.a ]]; then
	cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION
	make -j 8
    make install 
fi 
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/json ..."

cd thirdparty
if [ ! -d json ]; then
    git clone https://github.com/nlohmann/json.git json
    #git fetch --all --tags # to fetch tags 
    cd json
    git checkout bc889afb4c5bf1c0d8ee29ef35eaaf4c8bef8a5d   # release/3.11.2' 
    cd .. 
fi
cd json
make_buid_dir
if [[ ! -d install ]]; then
	cd build
    JSON_OPTIONS="-DJSON_BuildTests=OFF"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $JSON_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/sesync ..."

cd thirdparty
if [ ! -d sesync ]; then
    sudo apt-get install build-essential cmake-gui libeigen3-dev liblapack-dev libblas-dev libsuitesparse-dev
    git clone https://github.com/david-m-rosen/SE-Sync sesync
    #git fetch --all --tags # to fetch tags 
    cd sesync
    git checkout 9b631b6b82d8aa7d32ab846412ae1c070412b7b6  
    git submodule init
    git submodule update 
    git apply ../sesync.patch 
    cd .. 
fi
cd sesync/C++
make_buid_dir
if [[ ! -f build/lib/libSESync.so ]]; then
	cd build
    SESYNC_OPTIONS="-DENABLE_VISUALIZATION=ON"
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../../install" -DCMAKE_BUILD_TYPE=Release $SESYNC_OPTIONS $EXTERNAL_OPTION
	make -j 8
    #make install # unfortunately, sesync does not have any install target!
fi
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/dbow3 ..."

cd thirdparty
if [ ! -d dbow3 ]; then
    git clone https://github.com/rmsalinas/DBow3.git dbow3
    #git fetch --all --tags # to fetch tags 
    cd dbow3
    git checkout c5ae539abddcef43ef64fa130555e2d521098369
    git apply ../dbow3.patch   
    cd .. 
fi
cd dbow3
make_buid_dir
if [[ ! -d install ]]; then
	cd build
    DBOW3_OPTIONS="-DUSE_CONTRIB=ON"  # If you have installed the OpenCV contrib_modules
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $DBOW3_OPTIONS $EXTERNAL_OPTION
	make -j 8
    make install 
fi
cd $SCRIPT_DIR


# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/obindex2 ..."

cd thirdparty
if [ ! -d obindex2 ]; then
	sudo apt-get install -y libboost-system-dev libboost-filesystem-dev
	git clone https://github.com/emiliofidalgo/obindex2.git obindex2
    cd obindex2
    git checkout c79b76b849baf8fcd8246302daabe74059f9ed1c
    git apply ../obindex2.patch 
    cd .. 
fi
cd obindex2/lib
make_buid_dir
if [[ ! -d build/libobindex2.a ]]; then
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION
    make -j 8
fi 
cd $SCRIPT_DIR

# ====================================================

print_blue '================================================'
print_blue "Configuring and building thirdparty/ibow-lcd ..."

cd thirdparty
if [ ! -d ibow-lcd ]; then
	sudo apt-get install -y libboost-system-dev libboost-filesystem-dev
	git clone https://github.com/emiliofidalgo/ibow-lcd.git ibow-lcd
    cd ibow-lcd
    git checkout 0804b1eec5db88b3d252c02617b6c95b91ca0a96
    git apply ../ibow-lcd.patch 
    cd .. 
fi
cd ibow-lcd
make_buid_dir
if [[ ! -d build/liblcdetector.a ]]; then
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release $EXTERNAL_OPTION
    make -j 8
fi 
cd $SCRIPT_DIR


# ====================================================

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
    cmake .. -DCMAKE_INSTALL_PREFIX="`pwd`/../install" -DCMAKE_BUILD_TYPE=Release  $EXTERNAL_OPTION -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -fPIC" 
	make -j 8
    make install 
fi 
cd $SCRIPT_DIR


# ====================================================

if [ $USE_TENSORRT -eq 1 ]; then
    print_blue '================================================'
    print_blue "Configuring and building thirdparty/tensorrtbuffer ..."

    cd thirdparty
    cd tensorrtbuffer
    make_buid_dir
    if [[ ! -d lib/libtensorrtbuffer.so ]]; then
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release  $EXTERNAL_OPTION
        make -j 8
    fi 
    cd $SCRIPT_DIR
fi 


# ====================================================


echo '================================================'