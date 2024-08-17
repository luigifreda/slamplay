# SuperPoint SuperGlue TensorRT

SuperPoint and SuperGlue with TensorRT. Deploy with C++.

This is a modified version of the repository https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT. 
The slamplay project provides TensorRT install scripts, better cmake integration in the full project with a separation of the tensorrt buffer library (see thirdparty/tensorrtbuffers).

**Note**: It takes some minutes to build TensorRT engine at the first call. 

## Convert model(Optional)
The converted model is already provided in the [weights](./weights) folder, if you are using the pretrained model officially provided by [SuperPoint and SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), you do not need to go through this step.
```bash
python convert2onnx/convert_superpoint_to_onnx.py --weight_file superpoint_pth_file_path --output_dir superpoint_onnx_file_dir
python convert2onnx/convert_superglue_to_onnx.py --weight_file superglue_pth_file_path --output_dir superglue_onnx_file_dir
```

## Build and run
```bash
# test on image pairs 10 times, the output image will be saved in the build dir
test_superpointglue_image  
# test on the folder with image sequence, output images will be saved in the param assigned dir
test_superpointglue_sequence 
```
The default image size param is 320x240, if you need to modify the image size in the config file, you should delete the old .engine file in the weights dir.

## Samples
```c++
#include "super_point.h"
#include "super_glue.h"

// read image
cv::Mat image0 = cv::imread("../image/image0.png", cv::IMREAD_GRAYSCALE);
cv::Mat image1 = cv::imread("../image/image1.png", cv::IMREAD_GRAYSCALE);

// read config from file
Configs configs("../config/config.yaml", "../weights/");

// create superpoint detector and superglue matcher
auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);

// build engine
superpoint->build();
superglue->build();

// infer superpoint
Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
superpoint->infer(image0, feature_points0);
superpoint->infer(image1, feature_points1)

// infer superglue
std::vector<cv::DMatch> superglue_matches;
superglue->matching_points(feature_points0, feature_points1, superglue_matches);
 
```

## Acknowledgements

[SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)    
[SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork)     
[TensorRT](https://github.com/NVIDIA/TensorRT)     
[AirVO](https://github.com/xukuanHIT/AirVO)     
