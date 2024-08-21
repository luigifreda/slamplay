#!/usr/bin/env bash

function print_blue(){
    printf "\033[34;1m"
    printf "$@ \n"
    printf "\033[0m"
}


# Convert onnx model to TensorRT engine model

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
echo "current dir: $SCRIPT_DIR"
cd $SCRIPT_DIR # this brings us in the actual used folder (not the symbolic one)

export TENSORRT_DIR="$SCRIPT_DIR/../../thirdparty/TensorRT"
export PATH="$TENSORRT_DIR"/bin:$PATH
export LD_LIBRARY_PATH="$TENSORRT_DIR"/lib:$LD_LIBRARY_PATH

onnx_dir="$SCRIPT_DIR"
onnx_models_dir="$SCRIPT_DIR"/models

# for onnx_file in "$onnx_dir"/*.onnx
# do
    # if [ -f "$onnx_file" ]; then
    #     base_name=$(basename "${onnx_file%.onnx}")
    #     if [ ! -f "$onnx_dir/${base_name}.engine" ]; then
    #         # Documentation here https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/trtexec
    #         trtexec --onnx="$onnx_file" --saveEngine="$onnx_dir/${base_name}.engine" --best --fp16
    #         echo "Exported $onnx_file to $onnx_dir/${base_name}.engine"
    #     else 
    #         echo "Already exported $onnx_file to $onnx_dir/${base_name}.engine"
    #     fi
    # fi
# done
# echo "Exported all ONNX files to TensorRT engines."


# NOTE1: each model is converted with its own profile 
# NOTE2: if you use the python script (instead of this export script) you get the following error: 
#   1: [stdArchiveReader.cpp::StdArchiveReader::32] Error Code 1: Serialization (Serialization assertion magicTagRead == kMAGIC_TAG failed.Magic tag does not match)
#   4: [runtime.cpp::deserializeCudaEngine::66] Error Code 4: Internal Error (Engine deserialization failed.)

onnx_file=$SCRIPT_DIR/models/sam_onnx_example.onnx
if [ -f "$onnx_file" ]; then
    base_name=$(basename "${onnx_file%.onnx}")
    if [ ! -f "$onnx_models_dir/${base_name}.engine" ]; then
        print_blue converting $onnx_file to engine file
        # Documentation here https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/trtexec
        # https://docs.nvidia.com/tao/tao-toolkit/text/trtexec_integration/index.html
        # Options taken from python code
        trtexec --onnx="$onnx_file" --saveEngine="$onnx_models_dir/${base_name}.engine" \
        --fp16 --allowGPUFallback --useCudaGraph \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1  \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1  \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1 
        # removed --explicitBatch
        if [ $? -eq 0 ]; then
            echo "Exported $onnx_file to $onnx_dir/${base_name}.engine"
        fi
    else 
        echo "Already exported $onnx_file to $onnx_dir/${base_name}.engine"
    fi
fi

onnx_file=$SCRIPT_DIR/models/vit_l_embedding.onnx
if [ -f "$onnx_file" ]; then
    base_name=$(basename "${onnx_file%.onnx}")
    if [ ! -f "$onnx_models_dir/${base_name}.engine" ]; then
        print_blue converting $onnx_file to engine file
        # Documentation here https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/trtexec
        # https://docs.nvidia.com/tao/tao-toolkit/text/trtexec_integration/index.html
        # Options taken from python code
        trtexec --onnx="$onnx_file" --saveEngine="$onnx_models_dir/${base_name}.engine" \
        --fp16 --allowGPUFallback --useCudaGraph
        # removed --explicitBatch
        if [ $? -eq 0 ]; then   
            echo "Exported $onnx_file to $onnx_dir/${base_name}.engine"
        fi
    else 
        echo "Already exported $onnx_file to $onnx_dir/${base_name}.engine"
    fi
fi

onnx_file=$SCRIPT_DIR/models/sam_h_decoder_onnx.onnx
if [ -f "$onnx_file" ]; then
    base_name=$(basename "${onnx_file%.onnx}")
    if [ ! -f "$onnx_models_dir/${base_name}.engine" ]; then
        print_blue converting $onnx_file to engine file
        # Documentation here https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/trtexec
        # https://docs.nvidia.com/tao/tao-toolkit/text/trtexec_integration/index.html
        # Options taken from python code
        trtexec --onnx="$onnx_file" --saveEngine="$onnx_models_dir/${base_name}.engine" \
        --fp16 --allowGPUFallback --useCudaGraph \
        --minShapes=image_embeddings:1x256x64x64,point_coords:1x2x2,point_labels:1x2,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2  \
        --optShapes=image_embeddings:1x256x64x64,point_coords:1x5x2,point_labels:1x5,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2  \
        --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1,orig_im_size:1x2         
        # removed --explicitBatch
        #--workspace=6000 --verbose
        if [ $? -eq 0 ]; then   
            echo "Exported $onnx_file to $onnx_dir/${base_name}.engine"
        fi
    else 
        echo "Already exported $onnx_file to $onnx_dir/${base_name}.engine"
    fi
fi