#!/usr/bin/env bash

echo "Installing main dependencies ..."

set -e

# generic
sudo apt-get update
sudo apt-get install -y build-essential cmake   
sudo apt-get install -y libeigen3-dev 
sudo apt-get install -y libopenni-dev libopenni2-dev libpcl-dev
sudo apt-get install -y libyaml-cpp-dev

sudo apt-get install -y libzstd-devel 

sudo apt-get install -y libfmt-dev

# pangolin
sudo apt-get install -y libglew-dev libglfw3 libglfw3-dev
sudo apt-get install -y ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev
sudo apt-get install -y libjpeg-dev libpng12-dev libtiff5-dev libopenexr-dev

# g2o 
sudo apt-get install -y libsuitesparse-dev

# google dev 
sudo apt-get install -y libgoogle-glog-dev
sudo apt-get install -y libprotobuf-dev protobuf-compiler

# pcl 
sudo apt-get install -y libpcl-dev pcl-tools 

# octomap 
sudo apt-get install -y liboctomap-dev octovis 

# rerun 
sudo apt-get install -y cargo 

echo "...done"
