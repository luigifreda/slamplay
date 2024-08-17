#!/usr/bin/env sh

# Convert onnx model to TensorRT engine model


SCRIPT_DIR=`pwd`
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
#echo "current dir: $SCRIPT_DIR"
cd $SCRIPT_DIR # this brings us in the actual used folder (not the symbolic one)

export TENSORRT_DIR="$SCRIPT_DIR/../../thirdparty/TensorRT"
export PATH="$TENSORRT_DIR"/bin:$PATH
export LD_LIBRARY_PATH="$TENSORRT_DIR"/lib:$LD_LIBRARY_PATH

onnx_dir="$SCRIPT_DIR"/trained_data
for onnx_file in "$onnx_dir"/*.onnx
do
    if [ -f "$onnx_file" ]; then
        base_name=$(basename "${onnx_file%.onnx}")
        if [ ! -f "$onnx_dir/${base_name}.engine" ]; then
            # Documentation here https://github.com/NVIDIA/TensorRT/tree/release/8.2/samples/trtexec
            trtexec --onnx="$onnx_file" --saveEngine="$onnx_dir/${base_name}.engine" #--best --fp16
            echo "Exported $onnx_file to $onnx_dir/${base_name}.engine"
        else 
            echo "Already exported $onnx_file to $onnx_dir/${base_name}.engine"
        fi
    fi
done
echo "Exported all ONNX files to TensorRT engines."
