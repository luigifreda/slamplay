#!/usr/bin/env python

# files="\
# https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_indoor.pth \
# https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superglue_outdoor.pth \
# https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/models/weights/superpoint_v1.pth \
# https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth" 


import os
import argparse
import multiprocessing

import gdown

# folder of this script 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = SCRIPT_DIR

def gdrive_download(*args, **kwargs):
    url = kwargs["url"]
    output = kwargs["path"]
    # check if outfolder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output):
        print(f'downloading {url} to {output}')
        gdown.download(url, output, fuzzy=True)
    else: 
        print(f'file already exists: {output}')

def download_data(*args, **kwargs):
    p = multiprocessing.Process(target=gdrive_download, args=args, kwargs=kwargs)
    p.start()
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="quiet", action="store_false")
    args = parser.parse_args()
    args.quiet

    processes = []

    # download the large models that we want to use
    print("downloading superpointglue models...")
        
    # let's download just the large models that we want to use 
    p = download_data(
        path=WEIGHTS_DIR + "/superglue_indoor.pth",
        url="https://drive.google.com/file/d/1INeTMsENylSlbN_7mOE7M3CLrfgtACuY/view?usp=drive_link",
    )
    processes.append(p)
    
    p = download_data(
        path=WEIGHTS_DIR + "/superglue_outdoor.pth",
        url="https://drive.google.com/file/d/1-X9yctW8Eu5ITvhxr4XEe1CWP3QRilCk/view?usp=drive_link",
    )
    processes.append(p)
    
    p = download_data(
        path=WEIGHTS_DIR + "/superpoint_v1.pth",
        url="https://drive.google.com/file/d/12CcBkLgi7tqcxJ1Ly3eYksZRfTuqaSTE/view?usp=drive_link",
    )
    processes.append(p)
        
    
    for p in processes:
        p.join()
        
    print("download of onnx files completed!")
    
    #print("now converting from onnx to tensorrt engine models...")
    
    # TODO: 
    # To convert use the scripts (tested and working)
    # data/superpointglue/convert2onnx/convert_superglue_to_onnx.py
    # data/superpointglue/convert2onnx/convert_superpoint_to_onnx.py
    # run the conversion script from pth to onnx
    #os.system(f"bash {SCRIPT_DIR}/...")

if __name__ == "__main__":
    main()
