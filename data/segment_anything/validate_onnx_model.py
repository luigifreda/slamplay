#!/usr/bin/env python
import sys
import onnx
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, help="Path to your ONNX model")
    args = parser.parse_args()
    
    onnx_model_filename = args.filename
    model = onnx.load(onnx_model_filename)
    onnx.checker.check_model(model, full_check=True)   