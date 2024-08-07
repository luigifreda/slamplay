# HFNet C++

This folder contains a C++ HFNet interface implementation adapted from: 
https://github.com/LiuLimingCode/HFNet_SLAM

You can test and compare HFNet keypoint extraction/matching with ORB extraction/matching by testing the two examples (check in the build folder for their binaries after you have built everything): 
- `frontend/features_dl/hfnet/test_and_compare_extractors.cpp`
- `frontend/features_dl/hfnet/test_and_compare_matchers.cpp`

As for place recognition based on BoW (Bag Of Words) approach, check the example: 
- `frontend/features_dl/hfnet/test_match_global_feats.cpp`

A related comparison test on loop closure is availabel with: 
- `loop_closure/test_compare_loop_detection_orb_hfnet.cpp`


As a python reference, you can also find `hfnet` python scripts in the folder `hfnet` (from https://github.com/LiuLimingCode/HFNet_SLAM).