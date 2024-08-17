#!/usr/bin/env bash

# install other dependencies
pip install opencv-python pycocotools matplotlib onnxruntime onnx torch torchvision timm

if [ ! -d "segment-anything" ]; then
    git clone git@github.com:facebookresearch/segment-anything.git
    cd segment-anything
    pip install -e .
fi 

# if [ ! -d "MobileSAM" ]; then 
#     git clone git@github.com:ChaoningZhang/MobileSAM.git
#     cd MobileSAM
#     pip install -e .
# fi 