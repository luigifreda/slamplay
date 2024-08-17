#!/usr/bin/env python

import os
import argparse
import multiprocessing

import gdown

# folder of this script 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")

def gdrive_download(*args, **kwargs):
    url = kwargs["url"]
    output = kwargs["path"]
    # check if outfolder exists or create it
    output_folder = os.path.dirname(output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(output):
        print(f'downloading {url} to {output}')
        gdown.download(url, output)
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
    print("downloading SAM models...")

    # FILE_NAME1="sam_vit_l_0b3195.pth" 
    # URL_MY_DRIVE_ID1="1SNbCD7z1cHwFH0DmA-KE5OmB6zFMcIDn" # full file https://drive.google.com/file/d/1SNbCD7z1cHwFH0DmA-KE5OmB6zFMcIDn/view?usp=drive_link

    # FILE_NAME2="sam_onnx_example.onnx"
    # URL_MY_DRIVE_ID2="1VgIuWXycaDcYpeP3UWA1feq62Zon-JIb" # full file https://drive.google.com/file/d/1VgIuWXycaDcYpeP3UWA1feq62Zon-JIb/view?usp=drive_link

    # FILE_NAME3="vit_l_embedding.onnx"
    # URL_MY_DRIVE_ID3="1nwKg-CmEj0njHP4aABxW-3PT2ZnewRYF" # full file https://drive.google.com/file/d/1nwKg-CmEj0njHP4aABxW-3PT2ZnewRYF/view?usp=drive_link

        
    # let's download just the large models that we want to use 
    p = download_data(
        path=MODELS_DIR + "/sam_onnx_example.onnx",
        url="https://drive.google.com/uc?id=1VgIuWXycaDcYpeP3UWA1feq62Zon-JIb",
    )
    processes.append(p)
    
    p = download_data(
        path=MODELS_DIR + "/vit_l_embedding.onnx",
        url="https://drive.google.com/uc?id=1nwKg-CmEj0njHP4aABxW-3PT2ZnewRYF",
    )
    processes.append(p)
    
    p = download_data(
        path=WEIGHTS_DIR + "/sam_vit_l_0b3195.pth",
        url="https://drive.google.com/uc?id=1SNbCD7z1cHwFH0DmA-KE5OmB6zFMcIDn",
    )
    processes.append(p)
        
    
    for p in processes:
        p.join()
        
    print("download of onnx files completed!")
    
    print("now converting from onnx to tensorrt engine models...")
    
    # run the script onnx_to_tensorrt.py
    os.system(f"bash {SCRIPT_DIR}/onnx_to_tensorrt.sh")

if __name__ == "__main__":
    main()
