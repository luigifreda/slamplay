#!/usr/bin/env bash

# clean all

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

build_folders=( \
thirdparty/tensorrtbuffers \
thirdparty/dbow2 \
)

for folder in "${build_folders[@]}"
do
	echo "cleaning build $folder ..."
	if [[ -d $folder ]]; then 
		cd $folder
		if [[ -d build ]]; then 
			cd build
			make clean
			cd ..
			rm -Rf build
		fi 
		if [[ -d install ]]; then
			rm -Rf install
		fi
		if [[ -d lib ]]; then
			rm -Rf lib
		fi
		cd $SCRIPT_DIR
	fi 
done
#remove symbolic links
if [ -e thirdparty/tensorrtbuffers ]; then
	rm -Rf thirdparty/tensorrtbuffers
fi 

if [ -d thirdparty/TensorRT ]; then
	rm -Rf thirdparty/TensorRT
fi 


# ==========================================

git_folders=( \
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
thirdparty/tracy \
)

for folder in "${git_folders[@]}"; do
	echo "cleaning git $folder ..."
	if [[ -d $folder ]]; then 
		cd $folder
		if [[ -d build ]]; then 
			rm -Rf build
		fi 
		if [[ -d install ]]; then
			rm -Rf install
		fi 
		if [[ -d lib ]]; then 
			rm -Rf lib
		fi 
		cd $SCRIPT_DIR
	fi 
	rm -Rf $folder 
done


echo "cleaning build ..."
if [[ -d build ]]; then 
	rm -Rf build 
fi 


