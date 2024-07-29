#!/usr/bin/env bash

# clean all

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

folders=( \
thirdparty/pangolin \
thirdparty/sophus \
thirdparty/ceres \
thirdparty/gtsam \
thirdparty/g2o \
thirdparty/mahigui \
thirdparty/json \
thirdparty/sesync \
thirdparty/obindex2 \
thirdparty/ibow-lcd \
thirdparty/dbow3 \
thirdparty/rerun \
)

for folder in "${folders[@]}"
do
	echo "cleaning $folder ..."
	if [[ -d $folder ]]; then 
		cd $folder
		if [[ -d build ]]; then 
			rm -Rf build
		fi 
		cd $SCRIPT_DIR
	fi 
	rm -Rf $folder 
done


echo "cleaning build ..."
if [[ -d build ]]; then 
	rm -Rf build 
fi 