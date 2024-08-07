# HF-Net Model

This project is based on https://github.com/ethz-asl/hfnet.

The original HF-Net project was built with TensorFlow 1. However, Google Colab removed support for TensorFlow 1. The C++ and Python versions of TensorFlow 1 are no longer updated and are different to install. Therefore, to increase the compatibility of the HF-Net model and convert the model for TensorRT, extra operations are needed.

## Convert model for yourself

### Step 1: TensorFlow 1 to TensorFlow 2

1. Install TensorFlow 1 Python

You must install TensorFlow<=1.15. Here is an easy way to install the correct version.

```
pip install --upgrade pip
pip install nvidia-pyindex
pip install nvidia-tensorflow[horovod]
pip install nvidia-tensorboard==1.15
```

2. Download the checkpoint files from [here](https://projects.asl.ethz.ch/datasets/doku.php?id=cvpr2019hfnet).

3. Export the model

```
python3 export_model.py path_to_checkpoint_dir path_to_output_dir
```

### Step 2: TensorFlow 2 to ONNX

1. install ONNX graphsurgeon

```
python3 -m pip install numpy onnx
pip3 install tf2onnx
sudo apt-get install onnx-graphsurgeon
```

2. Export the model

```
conda activate tensorflow-1
python -m tf2onnx.convert --saved-model path_to_input_dir --output path_to_output_dir/HF-Net.onnx --inputs image:0 --outputs scores_dense_nms:0,local_descriptor_map:0,global_descriptor:0
```

## What's the differences?

In Step 1: The original HF-Net needs the support of tf.contrib.resampler, which is no longer supported. Therefore, the resampler model is removed and re-implemented in the BasedModel.cc file. Besides, To improve the efficiency, the number of NMS iterations is reduced from 3 to 2.

In Step 2: The TensorFlow saved_model cannot be read by the TensorRT engine, it should be convert to ONNX Model.
