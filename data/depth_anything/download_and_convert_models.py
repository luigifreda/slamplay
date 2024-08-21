#!/usr/bin/env python

import os
import argparse
import multiprocessing

import gdown

# folder of this script 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_DIR = os.path.join(SCRIPT_DIR, "trained_data")


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
    print("downloading DepthAnything v2 onnx models...")

    # p = download_data(
    #     path=TARGET_DIR + "/depth_anything_v2_metric_hypersim_vits.onnx",
    #     url="https://drive.google.com/uc?id=1dPpKkQR0kGyjNyjrJR6BnwgZ9WZjz6U_",
    # )
    # processes.append(p)
    
    # p= download_data(
    #     path=TARGET_DIR + "/depth_anything_v2_metric_hypersim_vitb.onnx",
    #     url="https://drive.google.com/uc?id=1TInjNSPRD_e1k6vm3RWLRbhYCz6xunVB",
    # )
    # processes.append(p)
        
        
    # let's download just the large models that we want to use 
    p = download_data(
        path=TARGET_DIR + "/depth_anything_v2_metric_hypersim_vitl.onnx",
        url="https://drive.google.com/uc?id=1KDRpHXaTlm2yyDzak08dXIDnEnqGUA72",
    )
    processes.append(p)
    
    
    p = download_data(
        path=TARGET_DIR + "/depth_anything_v2_metric_vkitti_vitl.onnx",
        url="https://drive.google.com/uc?id=12t5EqUY854OyRM2HCevVuTl7FJwZdCU8",
    )
    processes.append(p)
    
    
    for p in processes:
        p.join()
        
    print("download of onnx files completed!")
    
    print("now converting from onnx to tensorrt engine models...")
    
    # run the script onnx_to_tensorrt.py
    print(f'running {SCRIPT_DIR}/onnx_to_tensorrt.sh')
    os.system(f"bash {SCRIPT_DIR}/onnx_to_tensorrt.sh")

if __name__ == "__main__":
    main()
