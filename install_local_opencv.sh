#!/usr/bin/env bash
# Author: Luigi Freda 

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used
cd $SCRIPT_DIR_ # this brings us in the actual folder of this config script (not the symbolic one)
#echo SCRIPT_DIR_: $SCRIPT_DIR_

# ====================================================

function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_red(){
	printf "\033[31;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_yellow(){
	printf "\033[33;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_green(){
	printf "\033[32;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function check_package(){
    package_name=$1
    PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $package_name |grep "install ok installed")
    #echo "checking for $package_name: $PKG_OK"
    if [ "" == "$PKG_OK" ]; then
      #echo "$package_name is not installed"
      echo 1
    else
      echo 0
    fi
}

function install_package(){
    do_install=$(check_package $1)
    if [ $do_install -eq 1 ] ; then
        sudo apt-get install -y $1
    fi 
}

function install_packages(){
    for var in "$@"
    do
        install_package "$var"
    done
}

function cudnn_dev_installed(){
    for pkg in libcudnn8-dev libcudnn-dev nvidia-cudnn; do
        if [ "$(check_package "$pkg")" -eq 0 ]; then
            return 0
        fi
    done
    if [ -f /usr/include/cudnn.h ] || [ -f /usr/include/x86_64-linux-gnu/cudnn.h ]; then
        return 0
    fi
    return 1
}

function install_cudnn_dev(){
    if cudnn_dev_installed; then
        print_green "cuDNN development files already present, skipping install"
        return 0
    fi
    # Ubuntu 24.04 maps libcudnn-dev -> nvidia-cudnn (multiverse), which conflicts with
    # libcudnn8-dev from the NVIDIA CUDA repo. Prefer the CUDA-repo packages when available.
    if [[ $version == *"24.04"* ]]; then
        if apt-cache show libcudnn8-dev &>/dev/null; then
            sudo apt-get install -y libcudnn8 libcudnn8-dev
        else
            print_yellow "libcudnn8-dev not in apt cache; trying nvidia-cudnn (Ubuntu multiverse)"
            sudo apt-get install -y nvidia-cudnn
        fi
    else
        sudo apt-get install -y libcudnn8 libcudnn8-dev
    fi
}

function get_usable_cuda_version(){
    version="$1"
    if [[ "$version" != *"cuda"* ]]; then
        version="cuda-${version}"      
    fi 
    # check if we have two dots in the version, check if the folder exists otherwise remove last dot
    if [[ $version =~ ^[a-zA-Z0-9-]+\.[0-9]+\.[0-9]+$ ]]; then
        if [ ! -d /usr/local/$version ]; then 
            version="${version%.*}"  # remove last dot        
        fi     
    fi    
    echo $version
}

# Helper function to find library in a directory (handles versioned .so files)
find_lib_in_dir() {
    local libname=$1
    local lib_dir=$2
    local found=$(find "$lib_dir" -maxdepth 1 -name "${libname}.so*" -type f 2>/dev/null | head -n 1)
    if [ -n "$found" ]; then
        echo "$found"
    else
        echo ""
    fi
}

# Resolve nvcc from CUDADIR, CUDA_VERSION, or PATH (must run after CUDA PATH is exported).
function cuda_nvcc_bin(){
    local candidate
    for candidate in \
        "${CUDADIR:+$CUDADIR/bin/nvcc}" \
        "${CUDA_VERSION:+/usr/local/$CUDA_VERSION/bin/nvcc}" \
        "$(command -v nvcc 2>/dev/null)"; do
        if [[ -n "$candidate" && -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Convert nvcc arch code to CUDA_ARCH_BIN form (75 -> 7.5, 100 -> 10.0, 120 -> 12.0).
function cuda_arch_code_to_decimal(){
    local arch="$1"
    local len=${#arch}
    if [[ $len -eq 2 ]]; then
        echo "${arch:0:1}.${arch:1:1}"
    elif [[ $len -ge 3 ]]; then
        echo "${arch:0:$((len - 1))}.${arch:$((len - 1)):1}"
    fi
}

function get_cuda_arch_bin(){
    # Detect CUDA compute architectures for OpenCV's CUDA_ARCH_BIN.
    local arch_bin="" nvcc_bin nvcc_output cuda_version_str cuda_root

    nvcc_bin=$(cuda_nvcc_bin) || true

    # Method 1: nvcc --list-gpu-arch (what the installed toolkit supports)
    if [[ -n "$nvcc_bin" ]]; then
        nvcc_output=$("$nvcc_bin" --list-gpu-arch 2>/dev/null)
        if [[ $? -eq 0 && -n "$nvcc_output" ]]; then
            while IFS= read -r line; do
                local arch="" decimal=""
                if [[ $line =~ compute_([0-9]+) ]]; then
                    arch="${BASH_REMATCH[1]}"
                elif [[ $line =~ sm_([0-9]+) ]]; then
                    arch="${BASH_REMATCH[1]}"
                fi
                if [[ -n "$arch" ]]; then
                    decimal=$(cuda_arch_code_to_decimal "$arch")
                    if [[ -n "$decimal" ]]; then
                        arch_bin="${arch_bin}${decimal} "
                    fi
                fi
            done <<< "$nvcc_output"
            arch_bin=$(echo "$arch_bin" | xargs)
            if [[ -n "$arch_bin" ]]; then
                echo "$arch_bin"
                return 0
            fi
        fi
    fi

    # Method 2: infer a reasonable arch list from the CUDA toolkit version
    cuda_version_str=""
    for cuda_root in "${CUDADIR:-}" "${CUDA_VERSION:+/usr/local/$CUDA_VERSION}" "/usr/local/cuda"; do
        [[ -z "$cuda_root" || ! -d "$cuda_root" ]] && continue
        if [[ -f "$cuda_root/version.json" ]]; then
            cuda_version_str=$(grep -m1 '"version"' "$cuda_root/version.json" | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
            break
        elif [[ -f "$cuda_root/version.txt" ]]; then
            cuda_version_str=$(cat "$cuda_root/version.txt" 2>/dev/null)
            break
        fi
    done
    if [[ -z "$cuda_version_str" && -n "$nvcc_bin" ]]; then
        cuda_version_str=$("$nvcc_bin" --version 2>/dev/null | grep "release" | awk '{print $5}' | sed 's/,//')
    fi
    
    if [ -n "$cuda_version_str" ]; then
        # Extract major.minor version
        local cuda_major=$(echo "$cuda_version_str" | sed 's/[^0-9]*\([0-9]*\)\.\([0-9]*\).*/\1/')
        local cuda_minor=$(echo "$cuda_version_str" | sed 's/[^0-9]*\([0-9]*\)\.\([0-9]*\).*/\2/')
        
        # Set architectures based on CUDA version compatibility
        if [ "$cuda_major" -ge 13 ]; then
            # CUDA 13.x: supports 7.5, 8.0, 8.6, 8.9, 9.0
            arch_bin="7.5 8.0 8.6 8.9"
        elif [ "$cuda_major" -eq 12 ]; then
            # CUDA 12.x: supports 5.0-9.0
            arch_bin="6.1 7.0 7.5 8.0 8.6 8.9"
        elif [ "$cuda_major" -eq 11 ]; then
            # CUDA 11.x: supports 3.5-8.6
            if [ "$cuda_minor" -ge 8 ]; then
                arch_bin="6.1 7.0 7.5 8.0 8.6"
            else
                arch_bin="6.1 7.0 7.5 8.0"
            fi
        elif [ "$cuda_major" -eq 10 ]; then
            # CUDA 10.x: supports 3.0-7.5
            arch_bin="6.1 7.0 7.5"
        else
            # CUDA 9.x and older: supports 3.0-7.0
            arch_bin="6.1 7.0"
        fi
        
        if [ -n "$arch_bin" ]; then
            echo "$arch_bin"
            return 0
        fi
    fi
    
    # Method 3: safe defaults for modern GPUs
    print_red "Warning: Could not detect CUDA architectures. Using default values for modern GPUs." >&2
    echo "7.5 8.0 8.6 8.9"
}

# Read KEY from the local .env file; prints nothing if missing (optional second arg: path).
function read_env_var(){
    local key="$1"
    local env_file="${2:-$(local_env_file)}"
    if [[ -z "$key" || ! -f "$env_file" ]]; then
        return 0
    fi
    local line value
    line=$(grep -E "^[[:space:]]*${key}=" "$env_file" 2>/dev/null | tail -n 1)
    if [[ -z "$line" ]]; then
        return 0
    fi
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
        value="${value:1:-1}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
        value="${value:1:-1}"
    fi
    printf '%s' "$value"
}

# ====================================================

export TARGET_FOLDER=thirdparty

export OPENCV_VERSION="4.13.0"   # OpenCV version to download and install. See tags in https://github.com/opencv/opencv 

# ====================================================
print_blue  "Configuring and building $TARGET_FOLDER/opencv ..."

set -e

STARTING_DIR=`pwd`
version=$(lsb_release -a 2>&1)  # ubuntu version 

if [ ! -d $TARGET_FOLDER ]; then 
    mkdir -p $TARGET_FOLDER
fi 

# set CUDA 
#export CUDA_VERSION="cuda-11.8"  # must be an installed CUDA path in /usr/local; 
                                  # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
CUDA_ON=ON
if [[ -f $SCRIPT_DIR_/.env ]]; then
    CUDA_VERSION=$(read_env_var "CUDA_VERSION")
    echo reading CUDA_VERSION from .env file
    echo CUDA_VERSION: $CUDA_VERSION
fi
if [[ -n "$CUDA_VERSION" ]]; then
    CUDA_VERSION=$(get_usable_cuda_version $CUDA_VERSION)
    echo using CUDA $CUDA_VERSION
	if [ ! -d /usr/local/$CUDA_VERSION ]; then 
		echo CUDA $CUDA_VERSION does not exist
		CUDA_ON=OFF
	fi 
else
    if [ -d /usr/local/cuda ]; then
        CUDA_VERSION="cuda"  # use last installed CUDA path 
        echo using CUDA $CUDA_VERSION        
    else
        print_red "Warning: CUDA $CUDA_VERSION not found and will not be used!"
        CUDA_ON=OFF
    fi 
fi 
echo CUDA_ON: $CUDA_ON

if [[ "$CUDA_ON" == "ON" ]]; then
    echo CUDA_PATH: /usr/local/$CUDA_VERSION
    export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi

# Detect CUDA architectures automatically (requires nvcc on PATH or under CUDA_VERSION)
CUDA_ARCH_BIN=""
if [[ "$CUDA_ON" == "ON" ]]; then
    print_blue "Detecting CUDA compute architectures..."
    CUDA_ARCH_BIN=$(get_cuda_arch_bin)
    echo "Detected CUDA_ARCH_BIN: $CUDA_ARCH_BIN"
fi 

WITH_DNN=ON                # this can be used to turn off the DNN module
WITH_PROTOBUF=ON           # this can be used to turn off the protobuf module (it is required for DNN)
WITH_APPLE_FRAMEWORK=OFF
if [[ "$version" == *"darwin"* ]]; then
    #WITH_APPLE_FRAMEWORK=ON   # this will make opencv generate a single libopencv_world.so without the separate modules
    CUDA_ON=OFF
    WITH_PROTOBUF=OFF  # I am getting a protobuf version error on my mac
fi


WITH_NEON=OFF
arch=$(uname -m)
if [[ "$arch" == "arm64" || "$arch" == "aarch64" || "$arch" == arm* ]]; then
    WITH_NEON=ON
fi

WITH_QT=ON  
WITH_GTK=$(if [ "$WITH_QT" == "ON" ]; then echo "OFF"; else echo "ON"; fi)
WITH_OPENGL=ON

# pre-installing some required packages 

export BUILD_SFM_OPTION="ON"
if [[ $version == *"24.04"* ]] ; then
    BUILD_SFM_OPTION="OFF"  # it seems this module brings some build issues with Ubuntu 24.04
fi

if [[ ! -d $TARGET_FOLDER/opencv ]]; then
	sudo apt-get update
	sudo apt-get install -y pkg-config libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev libeigen3-dev libtbb-dev libgtk2.0-dev libopenblas-dev
    sudo apt-get install -y curl software-properties-common unzip
    sudo apt-get install -y build-essential cmake 
    if [[ "$CUDA_ON" == "ON" ]]; then
        install_cudnn_dev
    fi

    if [[ $version == *"22.04"* || $version == *"24.04"* ]] ; then
        sudo apt install -y libtbb-dev libeigen3-dev 
        sudo apt install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev 
        sudo add-apt-repository -y "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 # for libjasper-dev 
        sudo apt update
        sudo apt install -y libjasper-dev
        sudo apt install -y libv4l-dev libdc1394-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
                                    libopencore-amrnb-dev libopencore-amrwb-dev libxine2-dev libva-dev           
    fi
    if [[ $version == *"20.04"* ]] ; then
        sudo apt install -y libtbb-dev libeigen3-dev 
        sudo apt install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev 
        sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
        sudo apt install -y libjasper-dev
        sudo apt install -y libv4l-dev libdc1394-22-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
                                libopencore-amrnb-dev libopencore-amrwb-dev libxine2-dev            
    fi        
    if [[ $version == *"18.04"* ]] ; then
        sudo apt-get install -y libpng-dev 
        sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
        sudo apt-get install -y libjasper-dev
    fi
    if [[ $version == *"16.04"* ]] ; then
        sudo apt-get install -y libpng12-dev libjasper-dev 
    fi        

	DO_INSTALL_FFMPEG=$(check_package ffmpeg)
	if [ $DO_INSTALL_FFMPEG -eq 1 ] ; then
		echo "installing ffmpeg and its dependencies"
		sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev 
	fi
fi

# now let's download and compile opencv and opencv_contrib
# N.B: if you want just to update cmake settings and recompile then remove "opencv/install" and "opencv/build/CMakeCache.txt"

cd $TARGET_FOLDER

# Choose correct core library name based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    OPENCV_CORE_LIB="opencv/install/lib/libopencv_core.dylib"
else
    OPENCV_CORE_LIB="opencv/install/lib/libopencv_core.so"
fi

if [ ! -f $OPENCV_CORE_LIB ]; then
    if [ ! -d opencv ]; then
      wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
      sleep 1
      unzip $OPENCV_VERSION.zip
      rm $OPENCV_VERSION.zip
      cd opencv-$OPENCV_VERSION

      wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
      sleep 1
      unzip $OPENCV_VERSION.zip
      rm $OPENCV_VERSION.zip

      cd ..
      mv opencv-$OPENCV_VERSION opencv
    fi
    echo "entering opencv"
    cd opencv
    mkdir -p build
    mkdir -p install
    cd build
    echo "I am in "$(pwd)
    machine="$(uname -m)"
    echo OS: $version
    if [[ "$machine" == "x86_64" || "$machine" == "x64" || "$version" == "darwin"* ]]; then
		# standard configuration 
        echo "building laptop/desktop config under $version"
        # as for the flags and consider this nice reference https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_FFMPEG=ON \
          -DWITH_QT=$WITH_QT \
          -DWITH_GTK=$WITH_GTK \
          -DWITH_OPENGL=$WITH_OPENGL \
          -DWITH_TBB=ON \
          -DWITH_V4L=ON \
          -DWITH_CUDA=$CUDA_ON \
          -DWITH_CUBLAS=$CUDA_ON \
          -DWITH_CUFFT=$CUDA_ON \
          -DCUDA_FAST_MATH=$CUDA_ON \
          -DWITH_CUDNN=$CUDA_ON \
          -DBUILD_opencv_dnn=$WITH_DNN \
          -DOPENCV_DNN_CUDA=$CUDA_ON \
          -DCUDA_ARCH_BIN="$CUDA_ARCH_BIN" \
          -DBUILD_opencv_cudacodec=OFF \
          -DENABLE_FAST_MATH=1 \
          -DBUILD_opencv_sfm=$BUILD_SFM_OPTION \
          -DBUILD_NEW_PYTHON_SUPPORT=ON \
          -DBUILD_DOCS=OFF \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DINSTALL_C_EXAMPLES=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_opencv_apps=OFF \
          -DOPENCV_ENABLE_NONFREE=ON \
          -DBUILD_opencv_java=OFF \
          -DBUILD_opencv_python3=ON \
          -Wno-deprecated-gpu-targets \
          -DBUILD_PROTOBUF=${WITH_PROTOBUF:-OFF} \
          -DAPPLE_FRAMEWORK=${WITH_APPLE_FRAMEWORK:-OFF} ..
    else
        # Nvidia Jetson aarch64
        echo "building NVIDIA Jetson config"
        # Use detected architectures, fallback to Jetson-specific 6.2 if detection failed
        JETSON_ARCH_BIN="${CUDA_ARCH_BIN:-6.2}"
        if [ -z "$CUDA_ARCH_BIN" ]; then
            echo "Using Jetson default architecture: 6.2"
        fi
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_QT=$WITH_QT \
          -DWITH_GTK=$WITH_GTK \
          -DWITH_OPENGL=$WITH_OPENGL \
          -DWITH_TBB=ON \
          -DWITH_V4L=ON \
          -DWITH_CUDA=ON \
          -DWITH_CUBLAS=ON \
          -DWITH_CUFFT=ON \
          -DCUDA_FAST_MATH=ON \
          -DCUDA_ARCH_BIN="$JETSON_ARCH_BIN" \
          -DCUDA_ARCH_PTX="" \
          -DBUILD_opencv_cudacodec=OFF \
          -DENABLE_NEON=ON \
          -DENABLE_FAST_MATH=ON \
          -DBUILD_NEW_PYTHON_SUPPORT=ON \
          -DBUILD_DOCS=OFF \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DINSTALL_C_EXAMPLES=OFF \
          -DBUILD_EXAMPLES=OFF \
          -Wno-deprecated-gpu-targets ..
    fi
    make -j8
    make install 
fi

cd $STARTING_DIR

echo "...done with opencv"

