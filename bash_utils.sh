#!/usr/bin/env bash

# a collection of bash utils 


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

function ubuntu_version_string(){
    if [[ -n "$UBUNTU_VERSION" ]]; then
        echo "$UBUNTU_VERSION"
    else
        lsb_release -a 2>&1
    fi
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
    local ubuntu_version
    ubuntu_version=$(ubuntu_version_string)
    # Ubuntu 24.04 maps libcudnn-dev -> nvidia-cudnn (multiverse), which conflicts with
    # libcudnn8-dev from the NVIDIA CUDA repo. Prefer the CUDA-repo packages when available.
    if [[ $ubuntu_version == *"24.04"* ]]; then
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

function make_dir(){
    if [ ! -d $1 ]; then
        mkdir $1
    fi
}

function make_buid_dir(){
    make_dir build
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

function check_pip_package(){
    package_name=$1
    PKG_OK=$(pip list |grep $package_name)
    #print_blue "checking for package $package_name: $PKG_OK"
    if [ "" == "$PKG_OK" ]; then
      #print_blue "$package_name is not installed"
      echo 1
    else
      echo 0
    fi
}
function install_pip_package(){
    do_install=$(check_pip_package $1)
    if [ $do_install -eq 1 ] ; then
        pip3 install --user $1
    fi 
}
function install_pip_packages(){
    for var in "$@"
    do
        install_pip_package "$var"
    done
}


function extract_version(){
    #version=$(echo $1 | sed 's/[^0-9]*//g')
    #version=$(echo $1 | sed 's/[[:alpha:]|(|[:space:]]//g')
    version=$(echo $1 | sed 's/[[:alpha:]|(|[:space:]]//g' | sed s/://g)
    echo $version
}

function ubuntu_release_version(){
    if [[ -n "${UBUNTU_VERSION:-}" ]]; then
        echo "$UBUNTU_VERSION" | grep -oE '[0-9]+\.[0-9]+' | head -1
        return 0
    fi
    lsb_release -rs 2>/dev/null || ubuntu_version_string | grep -oE '[0-9]+\.[0-9]+' | head -1
}

# Map cuda-11.8.0 -> cuda-11.8 or cuda-11 when the versioned folder is missing.
function get_usable_cuda_version(){
    local version="$1"
    if [[ "$version" != *"cuda"* ]]; then
        version="cuda-${version}"
    fi
    if [[ $version =~ ^[a-zA-Z0-9-]+\.[0-9]+\.[0-9]+$ ]]; then
        if [ ! -d "/usr/local/$version" ]; then
            version="${version%.*}"
        fi
    fi
    echo "$version"
}

# Newest installed CUDA toolkit under /usr/local (e.g. cuda-12.6 or cuda symlink target).
function detect_available_cuda_dir(){
    local dir candidate best="" best_ver=""
    if [[ -d /usr/local/cuda/bin && -x /usr/local/cuda/bin/nvcc ]]; then
        if [[ -L /usr/local/cuda ]]; then
            dir=$(readlink -f /usr/local/cuda)
            candidate=$(basename "$dir")
            if [[ -d "/usr/local/$candidate/bin" ]]; then
                echo "$candidate"
                return 0
            fi
        fi
        echo "cuda"
        return 0
    fi
    for dir in /usr/local/cuda-*; do
        [[ -d "$dir/bin" && -x "$dir/bin/nvcc" ]] || continue
        candidate=$(basename "$dir")
        local ver="${candidate#cuda-}"
        if [[ -z "$best" ]] || [[ "$(printf '%s\n%s\n' "$ver" "$best_ver" | sort -V | tail -1)" == "$ver" ]]; then
            best="$candidate"
            best_ver="$ver"
        fi
    done
    if [[ -n "$best" ]]; then
        echo "$best"
        return 0
    fi
    return 1
}

# Ubuntu 20.04: cuda-11.8. Ubuntu 24.04: auto-detect. Others: env/config then detect.
function resolve_cuda_version_dir(){
    local ubuntu resolved
    ubuntu=$(ubuntu_release_version)
    if [[ "$ubuntu" == "20.04" ]]; then
        for candidate in cuda-11.8 cuda-11; do
            if [[ -d "/usr/local/$candidate/bin" ]]; then
                get_usable_cuda_version "$candidate"
                return 0
            fi
        done
        print_red "Ubuntu 20.04: expected /usr/local/cuda-11.8 (or cuda-11) but none found"
        return 1
    fi
    if [[ "$ubuntu" == "24.04" ]]; then
        if resolved=$(detect_available_cuda_dir); [[ -n "$resolved" ]]; then
            echo "$resolved"
            return 0
        fi
        print_red "Ubuntu 24.04: no CUDA toolkit found under /usr/local"
        return 1
    fi
    if [[ -n "${CUDA_VERSION:-}" && -d "/usr/local/${CUDA_VERSION}/bin" ]]; then
        get_usable_cuda_version "$CUDA_VERSION"
        return 0
    fi
    detect_available_cuda_dir
}

function get_cuda_toolkit_root(){
    if [[ -n "$CUDADIR" && -d "$CUDADIR" ]]; then
        echo "$CUDADIR"
    elif [[ -n "$CUDA_VERSION" && -d "/usr/local/$CUDA_VERSION" ]]; then
        echo "/usr/local/$CUDA_VERSION"
    elif [ -d /usr/local/cuda ]; then
        echo "/usr/local/cuda"
    else
        echo ""
    fi
}

function get_cuda_version(){
    local cuda_root
    cuda_root=$(get_cuda_toolkit_root)
    if [[ -z "$cuda_root" ]]; then
        echo 0
        return
    fi
    if [ -f "$cuda_root/version.json" ]; then
        local ver
        ver=$(grep -m1 '"version"' "$cuda_root/version.json" | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
        if [[ -n "$ver" ]]; then
            echo "${ver%%.*}."$(echo "${ver#*.}" | cut -d. -f1)
            return
        fi
    fi
    if [ -f "$cuda_root/version.txt" ]; then
        CUDA_STRING=$(cat "$cuda_root/version.txt")
        extract_version "$CUDA_STRING"
        return
    fi
    if [ -x "$cuda_root/bin/nvcc" ]; then
        "$cuda_root/bin/nvcc" --version | grep release | awk '{print $5}' | sed 's/,//'
        return
    fi
    echo 0
}

function cuda_toolkit_root_for_version(){
    local ver="$1"
    local candidate
    for candidate in "cuda-${ver}" "cuda-$(echo "${ver}" | cut -d. -f1)" "cuda"; do
        if [[ -d "/usr/local/${candidate}/bin" ]]; then
            echo "/usr/local/${candidate}"
            return 0
        fi
    done
    echo ""
}

# OpenCV records the CUDA version used at build time; consumers must use the same toolkit.
function opencv_cuda_toolkit_cmake_flags(){
    local opencv_dir="${1:-${OpenCV_DIR:-}}"
    if [[ -z "$opencv_dir" ]]; then
        return 0
    fi
    local opencv_config="${opencv_dir%/}/OpenCVConfig.cmake"
    if [[ ! -f "$opencv_config" ]]; then
        return 0
    fi
    local cuda_ver
    cuda_ver=$(grep '^set(OpenCV_CUDA_VERSION' "$opencv_config" | sed -E 's/.*"([^"]+)".*/\1/')
    if [[ -z "$cuda_ver" ]]; then
        return 0
    fi
    local toolkit_root
    toolkit_root=$(cuda_toolkit_root_for_version "$cuda_ver")
    if [[ -z "$toolkit_root" ]]; then
        print_yellow "OpenCV was built with CUDA ${cuda_ver} but no matching toolkit found under /usr/local"
        return 0
    fi
    echo "-DCUDA_TOOLKIT_ROOT_DIR=${toolkit_root}"
}

# CUDA 11.x officially supports GCC <= 11. On Ubuntu 24.04 the system compiler is newer;
# use -allow-unsupported-compiler so nvcc and C/C++ share one libstdc++ ABI (required for
# linking distro libs such as libglog and for Ceres unit tests).
# Not used on Ubuntu 20.04 (GCC 9/10) where cuda-11.8 is the supported pairing.
function cuda_nvcc_compat_cmake_flags(){
    local nvcc="${1:-}"
    if [[ -z "$nvcc" ]]; then
        nvcc=$(command -v nvcc)
    fi
    if [[ -z "$nvcc" || ! -x "$nvcc" ]]; then
        return 0
    fi
    local cuda_release cuda_major
    cuda_release=$("$nvcc" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || true)
    cuda_major="${cuda_release%%.*}"
    if [[ -z "$cuda_major" || "$cuda_major" -gt 11 ]]; then
        return 0
    fi
    local gcc_major
    gcc_major=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
    if [[ -n "$gcc_major" && "$gcc_major" -gt 11 ]]; then
        echo "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    fi
}

# Backward-compatible alias
function cuda_host_compiler_cmake_flags(){
    cuda_nvcc_compat_cmake_flags "$@"
}

function get_current_nvidia_driver_version(){
    NVIDIA_DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
    echo $NVIDIA_DRIVER_VERSION
}

# Default .env path: CONFIG_DIR/.env when config.sh was sourced, else ./.env
function local_env_file(){
    if [[ -n "${CONFIG_DIR:-}" ]]; then
        echo "${CONFIG_DIR}/.env"
    else
        echo ".env"
    fi
}

# Write or update KEY=VALUE in the local .env file (optional third arg: path).
function write_env_var(){
    local key="$1"
    local value="$2"
    local env_file="${3:-$(local_env_file)}"
    if [[ -z "$key" ]]; then
        print_red "write_env_var: key is required"
        return 1
    fi
    local dir found=0 line
    dir=$(dirname "$env_file")
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
    fi
    if [[ -f "$env_file" ]]; then
        : > "${env_file}.tmp"
        while IFS= read -r line || [[ -n "$line" ]]; do
            if [[ "$line" =~ ^[[:space:]]*${key}= ]]; then
                printf '%s=%s\n' "$key" "$value" >> "${env_file}.tmp"
                found=1
            else
                printf '%s\n' "$line" >> "${env_file}.tmp"
            fi
        done < "$env_file"
        if [[ $found -eq 0 ]]; then
            printf '%s=%s\n' "$key" "$value" >> "${env_file}.tmp"
        fi
        mv "${env_file}.tmp" "$env_file"
    else
        printf '%s=%s\n' "$key" "$value" > "$env_file"
    fi
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

function gdrive_download () {
  if gdown -V >/dev/null 2>&1; then
    echo "" #"gdown is found in PATH"
  else
    if [[ -f $HOME/.local/bin/gdown ]]; then
      export PATH=$HOME/.local/bin:$PATH
    fi 
  fi  
  gdown https://drive.google.com/uc?id=$1
}