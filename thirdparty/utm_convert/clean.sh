#!/usr/bin/env bash

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

cd $SCRIPT_DIR_

clean_folders=( \
build \
install \
lib \
include \
)

for folder in "${clean_folders[@]}"; do
    echo "cleaning $folder ..."
    if [ -d $folder ]; then
        rm -Rf $folder
    else
        echo "folder $folder does not exist"
    fi
done
