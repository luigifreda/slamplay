#!/usr/bin/env python

import argparse
import os

import numpy as np
import onnx
import torch

import superglue


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def reduce_l2(desc):
    dn = np.linalg.norm(desc, ord=2, axis=1)  # Compute the norm.
    desc = desc / np.expand_dims(dn, 1)  # Divide by norm to normalize.
    return desc


def main():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="weights/superglue_outdoor.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="output", help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    # load model
    superpoint_model = superglue.SuperGlue()
    pytorch_total_params = sum(p.numel() for p in superpoint_model.parameters())
    print('total number of params: ', pytorch_total_params)

    # initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    superpoint_model.load_state_dict(torch.load(weight_file, map_location=map_location))
    superpoint_model.eval()

    # create input to the model for onnx trace
    x0 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y0 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts0 = torch.stack((x0, y0), 2).float()
    scores0 = torch.randn(1, 512)
    desc0 = torch.randn(1, 256, 512)
    x1 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y1 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts1 = torch.stack((x1, y1), 2).float()
    scores1 = torch.randn(1, 512)
    desc1 = torch.randn(1, 256, 512)
    onnx_filename = os.path.join(output_dir,
                                 weight_file.split("/")[-1].split(".")[0] + ".onnx")

    # Export the model
    torch.onnx.export(superpoint_model,  # model being run
                      (kpts0, scores0, desc0, kpts1, scores1, desc1),  # model input (or a tuple for multiple inputs)
                      onnx_filename,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=["keypoints_0",  # batch x feature_number0 x 2
                                   "scores_0",  # batch x feature_number0
                                   "descriptors_0",  # batch x feature_dims x feature_number0
                                   "keypoints_1",  # batch x feature_number1 x 2
                                   "scores_1",  # batch x feature_number1
                                   "descriptors_1",  # batch x feature_dims x feature_number1
                                  ],  # the model input names
                      output_names=["scores"],  # the model output names
                      dynamic_axes={'keypoints_0': {1: 'feature_number_0'},
                                    'scores_0': {1: 'feature_number_0'},
                                    'descriptors_0': {2: 'feature_number_0'},
                                    'keypoints_1': {1: 'feature_number_1'},
                                    'scores_1': {1: 'feature_number_1'},
                                    'descriptors_1': {2: 'feature_number_1'},
                                    }  # dynamic model input names
                      )

    # check onnx model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    print("Exported model has been checked with ONNXRuntime.")


if __name__ == '__main__':
    main()
