#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

. config.sh  # source configuration file and utils

# NOTE: The used default tensorflow configuration in the downloaded tensorflow_cc project is: <tensorflow 2.9.0, bazel 5.1.1, cudnn 8.6.0, cuda 11.6>.
# If you configured slamplay to use a different version of cuda that may be a problem since (1) in general, you cannot mix/link different versions 
# of the same library in the same project, (2) other versions of cuda may not be compatible with the default tensorflow_cc configuration.
# In particular, tensorflow_cc allows to build newer tensorflow configurations (see the tested configurations https://www.tensorflow.org/install/source#gpu). 
# However, note that tensorflow does download and use its own custom versions of Eigen (and of other base libraries, according to the selected tensorflow version) and 
# these library versions may not be the same installed in your system. This fact may cause severe problems (undefined behaviors and uncontrolled crashes) 
# in your final target projects (where you want to import the built and deployed Tensorflow C++): In fact, you may be mixing libraries built with different 
# versions of Eigen (so with different data alignments)!

if [[ ! -d thirdparty ]]; then
    mkdir thirdparty
fi

cd thirdparty

# The default install path of tensorflow_cc is $HOME/.tensorflow 

if [[ ! -d $HOME/.tensorflow ]]; then
    if [[ ! -d tensorflow_cc ]]; then
        git clone https://github.com/luigifreda/tensorflow_cc
    fi 
    cd tensorflow_cc
    $ ./build.sh   # will use the default configuration reported in the main README file
fi 