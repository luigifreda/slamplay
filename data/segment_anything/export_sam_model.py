#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.nn import functional as F
from segment_anything.modeling import Sam   # install SAM as explained here https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation
from torchvision.transforms.functional import resize, to_pil_image
from typing import Tuple
from segment_anything import sam_model_registry, SamPredictor # install SAM as explained here https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#installation
import cv2
import matplotlib.pyplot as plt
import warnings
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import onnxruntime
import argparse
from pathlib import Path


# folder of this script 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEST_IMAGE = os.path.join(SCRIPT_DIR, "truck.jpg")
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "weights")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

# @torch.no_grad()
def pre_processing(image: np.ndarray, target_length: int, device,pixel_mean,pixel_std,img_size):
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    input_image = np.array(resize(to_pil_image(image), target_size))
    input_image_torch = torch.as_tensor(input_image, device=device)
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Normalize colors
    input_image_torch = (input_image_torch - pixel_mean) / pixel_std

    # Pad
    h, w = input_image_torch.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))
    return input_image_torch

def export_embedding_model(image_path=DEFAULT_TEST_IMAGE):
    sam_checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_l_0b3195.pth")
    model_type = "vit_l"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # image_encoder = EmbeddingOnnxModel(sam)
    image = cv2.imread(image_path)
    print('input image shape: ', image.shape)
    
    target_length = sam.image_encoder.img_size  # should be 1014
    pixel_mean = sam.pixel_mean 
    pixel_std = sam.pixel_std
    img_size = sam.image_encoder.img_size
    
    print('target_length from model: ', target_length)
    print('image size from model: ', img_size)
    
    inputs = pre_processing(image, target_length, device, pixel_mean, pixel_std, img_size)
    onnx_model_path = os.path.join(MODELS_DIR, model_type+"_"+"embedding.onnx")
    dummy_inputs = {
        "images": inputs
    }
    output_names = ["image_embeddings"]
    image_embeddings = sam.image_encoder(inputs).cpu().numpy()
    print('image_embeddings', image_embeddings.shape)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                sam.image_encoder,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                # dynamic_axes=dynamic_axes,
            )    

def export_sam_model(image_path=DEFAULT_TEST_IMAGE):
    image = cv2.imread(image_path)
        
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_l_0b3195.pth")
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model_path = os.path.join(MODELS_DIR,"sam_onnx_example.onnx")

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    target_length = 1024 
    input_image_size = image.shape[:2] #[1500, 2250]
    print('input image size: ', input_image_size)
    image_size = get_preprocess_shape(input_image_size[0], input_image_size[1], target_length)
    #image_size = image.shape[:2]
    #image_size = [1500, 2250]
    print('preprocessed image size: ', image_size)

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor(image_size, dtype=torch.int32),
    }
    # output_names = ["masks", "iou_predictions", "low_res_masks"]
    output_names = ["masks", "scores"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                #dynamic_axes=dynamic_axes,
            )  

def run_sam_model_example(image_path=DEFAULT_TEST_IMAGE):
    ort_session_embedding = onnxruntime.InferenceSession(os.path.join(MODELS_DIR,'vit_l_embedding.onnx'),providers=['CPUExecutionProvider'])
    ort_session_sam = onnxruntime.InferenceSession(os.path.join(MODELS_DIR,'sam_onnx_example.onnx'),providers=['CPUExecutionProvider'])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image2 = image.copy()
    
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    target_length = image_size
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    device = "cpu"
    inputs = pre_processing(image, target_length, device, pixel_mean, pixel_std, image_size)
    ort_inputs = {
        "images": inputs.cpu().numpy()
    }
    image_embeddings = ort_session_embedding.run(None, ort_inputs)[0]

    #input_point = np.array([[500, 375]])
    input_point = np.array([[784, 379]])    
    input_label = np.array([1])

    #onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    #onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)
    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1,-1, -1, -1])], axis=0)[None, :].astype(np.float32)
    
        
    from segment_anything.utils.transforms import ResizeLongestSide
    transf = ResizeLongestSide(image_size)
    onnx_coord = transf.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    preproc_image_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)

    ort_inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(preproc_image_size, dtype=np.int32)
    }

    masks = ort_session_sam.run(None, ort_inputs)

    # If the masks_array has inconsistent shapes, try selecting the relevant part:
    # For example, if the output has extra dimensions, you may need to squeeze or index it
    if len(masks) >= 3:
        for i in range(len(masks)):
            print(f'masks {i} shape: {masks[i].shape}')
        masks = masks[0]  # Adjust this indexing based on your output shape

    masks_array = np.array(masks)    
    print(f'masks array shape: {masks_array.shape}')    
    
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_l_0b3195.pth")
    model_type = "vit_l"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model_path = os.path.join(MODELS_DIR,"sam_onnx_example.onnx")

    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    masks = onnx_model.mask_postprocessing(torch.as_tensor(masks_array), torch.as_tensor(image.shape[:2]))
    masks = masks > 0.0
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    # show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig('demo.png')
    plt.show()    

def export_engine_sam_image_encoder(f=os.path.join(MODELS_DIR,'vit_l_embedding.onnx')):
    import tensorrt as trt
    file = Path(f)
    f = file.with_suffix('.engine')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 6
    print("workspace: ", workspace)
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    half = True
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())


def export_engine_sam_sample_encoder_and_mask_decoder(f=os.path.join(MODELS_DIR,'sam_onnx_example.onnx')):
    import tensorrt as trt
    file = Path(f)
    f = file.with_suffix('.engine')  # TensorRT engine file
    onnx = file.with_suffix('.onnx')
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 10
    print("workspace: ", workspace)
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f'failed to load ONNX file: {onnx}')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'output "{out.name}" with shape{out.shape} {out.dtype}')

    profile = builder.create_optimization_profile()
    profile.set_shape('image_embeddings', (1, 256, 64, 64), (1, 256, 64, 64), (1, 256, 64, 64))
    profile.set_shape('point_coords', (1, 2,2), (1, 5,2), (1,10,2))
    profile.set_shape('point_labels', (1, 2), (1, 5), (1,10))
    profile.set_shape('mask_input', (1, 1, 256, 256), (1, 1, 256, 256), (1, 1, 256, 256))
    profile.set_shape('has_mask_input', (1,), (1, ), (1, ))
    # # profile.set_shape_input('orig_im_size', (416,416), (1024,1024), (1500, 2250))
    # profile.set_shape_input('orig_im_size', (2,), (2,), (2, ))
    config.add_optimization_profile(profile)

    half = True
    print(f'building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}')
    if builder.platform_has_fast_fp16 and half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(f, 'wb') as t:
        t.write(engine.serialize())


def collect_stats(model, device, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (path, im, im0s, vid_cap, s) in tqdm(enumerate(data_loader), total=num_batches):
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        model(im)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)

class embedding_model_part_1(nn.Module):
    def __init__(self) :
        super().__init__()
        sam_checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        device = "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

    def forward(self, x):
        x = self.sam.image_encoder.patch_embed(x)
        if self.sam.image_encoder.pos_embed is not None:
            x = x + self.sam.image_encoder.pos_embed

        val_1 = len(self.sam.image_encoder.blocks) // 2
        for blk in self.sam.image_encoder.blocks[0:val_1]:
            x = blk(x)

        return x
    
class embedding_model_part_2(nn.Module):
    def __init__(self) :
        super().__init__()
        sam_checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        device = "cpu"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)

    def forward(self, x):
        val_1 = len(self.sam.image_encoder.blocks) // 2
        for blk in self.sam.image_encoder.blocks[val_1:]:
            x = blk(x)
        x = self.sam.image_encoder.neck(x.permute(0, 3, 1, 2))
        return x

def export_embedding_model_part_1(image_path=DEFAULT_TEST_IMAGE):
    device = "cpu"
    model_type = "vit_h"
    model = embedding_model_part_1()
    # image_encoder = EmbeddingOnnxModel(sam)
    image = cv2.imread(image_path)
    target_length = model.sam.image_encoder.img_size
    print('target_length from model: ', target_length)
    pixel_mean = model.sam.pixel_mean 
    pixel_std = model.sam.pixel_std
    img_size = model.sam.image_encoder.img_size
    inputs = pre_processing(image, target_length, device,pixel_mean,pixel_std,img_size)
    onnx_model_path = os.path.join(MODELS_DIR, model_type+"_"+"part_1_embedding.onnx")
    dummy_inputs = {
        "images": inputs
    }
    output_names = ["image_embeddings_part_1"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                # dynamic_axes=dynamic_axes,
            )   

def export_embedding_model_part_2():
    device = "cpu"
    model_type = "vit_h"
    model = embedding_model_part_2()
    
    inputs = torch.randn(1, 64, 64, 1280)
    onnx_model_path = os.path.join(MODELS_DIR, model_type+"_"+"part_2_embedding.onnx")
    dummy_inputs = {
        "image_embeddings_part_1": inputs
    }
    output_names = ["image_embeddings_part_2"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                # dynamic_axes=dynamic_axes,
            )   
        
def export_sam_h_model(image_path=DEFAULT_TEST_IMAGE):
    image = cv2.imread(image_path)
    
    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    onnx_model_path = os.path.join(MODELS_DIR,"sam_h_decoder_onnx.onnx")

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    target_length = 1024 
    input_image_size = image.shape[:2] #[1500, 2250]
    print('input image size: ', input_image_size)
    image_size = get_preprocess_shape(input_image_size[0], input_image_size[1], target_length)
    #image_size = image.shape[:2]
    #image_size = [1500, 2250]
    print('preprocessed image size: ', image_size)

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor(image_size, dtype=torch.int32),
    }
    # output_names = ["masks", "iou_predictions", "low_res_masks"]
    output_names = ["masks", "scores"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=13,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                #dynamic_axes=dynamic_axes,
            )  

def run_sam_h_model_example(image_path=DEFAULT_TEST_IMAGE):
    import os
    ort_session_embedding_part_1 = onnxruntime.InferenceSession(os.path.join(MODELS_DIR,'vit_h_part_1_embedding.onnx'),providers=['CPUExecutionProvider'])
    ort_session_embedding_part_2 = onnxruntime.InferenceSession(os.path.join(MODELS_DIR,'vit_h_part_2_embedding.onnx'),providers=['CPUExecutionProvider'])
    ort_session_sam = onnxruntime.InferenceSession(os.path.join(MODELS_DIR,'sam_h_decoder_onnx.onnx'),providers=['CPUExecutionProvider'])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'image shape: {image.shape}')
    #image2 = image.copy()
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    target_length = image_size
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375]
    pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1)
    pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1)
    device = "cpu"
    inputs = pre_processing(image, target_length, device, pixel_mean,pixel_std,image_size)
    ort_inputs = {
        "images": inputs.cpu().numpy()
    }
    image_embeddings = ort_session_embedding_part_1.run(None, ort_inputs)[0]
    ort_inputs = {
        "image_embeddings_part_1": image_embeddings
    }
    image_embeddings = ort_session_embedding_part_2.run(None, ort_inputs)[0]

    input_point = np.array([[784, 379]])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0],[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1,-1, -1, -1])], axis=0)[None, :].astype(np.float32)
    
    from segment_anything.utils.transforms import ResizeLongestSide
    transf = ResizeLongestSide(image_size)
    onnx_coord = transf.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int32)
    }

    #masks, _ = ort_session_sam.run(None, ort_inputs)
    masks = ort_session_sam.run(None, ort_inputs)
    
    # If the masks_array has inconsistent shapes, try selecting the relevant part:
    # For example, if the output has extra dimensions, you may need to squeeze or index it
    if len(masks) >= 3:
        for i in range(len(masks)):
            print(f'masks {i} shape: {masks[i].shape}')
        masks = masks[0]  # Adjust this indexing based on your output shape

    masks_array = np.array(masks)    
    print(f'masks array shape: {masks_array.shape}')    

    from segment_anything.utils.onnx import SamOnnxModel
    checkpoint = os.path.join(WEIGHTS_DIR,"sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    onnx_model = SamOnnxModel(sam, return_single_mask=True)
    masks = onnx_model.mask_postprocessing(torch.as_tensor(masks_array), torch.as_tensor(image.shape[:2]))
    masks = masks > 0.0
    print(f'masks shape: {masks.shape}')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    # show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.savefig('demo.png')
    plt.show()


if __name__ == '__main__':
    # get options with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=DEFAULT_TEST_IMAGE, help='path to input image')
    parser.add_argument('--export_embedding_model', action='store_true')
    parser.add_argument('--export_sam_model', action='store_true')
    parser.add_argument('--run_sam_model_example', action='store_true')
    parser.add_argument('--export_engine_sam_image_encoder', action='store_true')
    parser.add_argument('--export_engine_sam_sample_encoder_and_mask_decoder', action='store_true')
    parser.add_argument('--export_embedding_model_part_1_and_2', action='store_true') 
    parser.add_argument('--export_sam_h_model', action='store_true')    
    parser.add_argument('--run_sam_h_model_example', action='store_true')    
    args = parser.parse_args()
    
    
    with torch.no_grad():
        if args.export_embedding_model:
            print('exporting embedding model')
            export_embedding_model(args.image_path)
        
        if args.export_sam_model:
            print('exporting sam model')
            export_sam_model(args.image_path)
        
        if args.run_sam_model_example:
            print('running sam model example')
            run_sam_model_example(args.image_path)
        
        if args.export_engine_sam_image_encoder:
            print('exporting engine image encoder')
            export_engine_sam_image_encoder()
        
        if args.export_engine_sam_sample_encoder_and_mask_decoder:
            print('exporting engine prompt encoder and mask decoder')
            export_engine_sam_sample_encoder_and_mask_decoder()
        
        if args.export_embedding_model_part_1_and_2:
            print('exporting embedding model part 1 and 2')
            export_embedding_model_part_1(args.image_path)
            export_embedding_model_part_2()
        
        if args.export_sam_h_model:
            print('exporting sam h model')
            export_sam_h_model(args.image_path)
        
        if args.run_sam_h_model_example:
            print('running sam h model example')
            run_sam_h_model_example(args.image_path)
        