# GPU support with `CUDA`, `cuDNN`, `TensorRT`

<!-- TOC -->

- [GPU support with CUDA, cuDNN, TensorRT](#gpu-support-with-cuda-cudnn-tensorrt)
    - [1. CUDA 11.8](#1-cuda-118)
    - [2. CUDA 12.5 - WIP](#2-cuda-125---wip)
    - [3. tensorflow_cc](#3-tensorflow_cc)
    - [4. Feedback](#4-feedback)

<!-- /TOC -->


I recommend the following tested configuration with `CUDA` ecosystem.

## CUDA 11.8 

- `CUDA` 11.8 (or 11.6). 
  * Install instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). 
- `cuDNN` 8.6.0.163-1+cuda11.8. 
  * Install instructions [here](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html).    
  * Once you downloaded the proper deb package, run the following command:         
`sudo apt install -y libcudnn8=8.6.0.163-1+cuda11.8 libcudnn8-dev=8.6.0.163-1+cuda11.8`       
- `TensorRT` 8.5.1.7 will be automatically installed by the main script `build.sh` (via `install_local_tensorrt.sh`) in `thirdparty/TensorRT` (once `CUDA` has been installed) and cmake will automatically use it by default.


## CUDA 12.5 - WIP

Tested via [rosdocker](https://github.com/luigifreda/rosdocker) (container `ubuntu24_cuda` with `CUDA` 12.5). Currently work in progress.

- `CUDA` 12.5
  * Install instructions [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html). 
- `cuDNN` 8.9.2.26~cuda12+3. 
  * Install instructions [here](https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html).    
  * Once you downloaded the proper deb package, run the following command:         
`sudo apt install -y nvidia-cudnn=8.9.2.26~cuda12+3`       
- `TensorRT` 10.3.0.26 will be automatically installed by the main script `build.sh` (via `install_local_tensorrt.sh`)in `thirdparty/TensorRT` (once `CUDA` has been installed) and cmake will automatically use it by default.


**Warning**: Currently, there are some known issues with: 
- SAM under `CUDA` 12.5. The behaviour is not exactly the same we have under `CUDA` 11.8.
- Superpointglue `CUDA` 12.5. The behaviour is not exactly the same we have under `CUDA` 11.8.  

## tensorflow_cc

As for tensorflow support, I recommend the usage of [tensorflow_cc](https://github.com/luigifreda/tensorflow_cc) and the following **tested configuration** (default one for [tensorflow_cc](https://github.com/luigifreda/tensorflow_cc)) under Ubuntu 20.04:
- **C++**: 17
- **TENSORFLOW_VERSION**: 2.9.0 
- **BAZEL_VERSION**: 5.1.1
- **CUDA**: 11.6 
- **CUDNN**: 8.6.0.163-1+cuda11.8       
  `sudo apt install -y libcudnn8=8.6.0.163-1+cuda11.8 libcudnn8-dev=8.6.0.163-1+cuda11.8`

As noted [here](https://github.com/luigifreda/tensorflow_cc?tab=readme-ov-file#some-final-notes-and-some-tested-working-configurations), I successfully built and deployed other newer tensorflow configurations (see the list [here](https://github.com/luigifreda/tensorflow_cc?tab=readme-ov-file#some-final-notes-and-some-tested-working-configurations)). However, note that tensorflow does download and use its own custom versions of Eigen (and of other base libraries, according to the selected tensorflow version) and the used library versions may not be the same ones that are installed in your system. This fact may cause severe issues (undefined behaviors and uncontrolled crashes) in your final target projects (where you import the built and deployed Tensorflow C++): In fact, in such case, you may be mixing libraries built with different versions of Eigen (so with different data alignments)!


## Feedback 

Feel free to share any issue and suggestions. I'll be very glad to receive any feedback and improve the install procedure.