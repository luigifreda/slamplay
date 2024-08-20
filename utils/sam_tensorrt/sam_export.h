#pragma once

#include <fstream>
#include <iostream>

#include "tensorrt/tensorrt_utils.h"

#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#define SET_MAX_WORKSPACE 1
#define MAX_WORKSPACE_IN_GB 6

inline void export_engine_sam_image_encoder(const std::string& f = "vit_l_embedding.onnx", const std::string& output = "vit_l_embedding.engine") {
    std::cout << "exporting " << f << " to " << output << std::endl;

    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    // auto parser = std::make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

#if SET_MAX_WORKSPACE
    // maximum workspace size
    const int workspace = MAX_WORKSPACE_IN_GB;  // GB
    config->setMaxWorkspaceSize(workspace * 1U << 30);
#endif
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
    std::cout << "serializedModel->size(): " << serializedModel->size() << std::endl;

    std::cout << "saving model to " << output << std::endl;
    std::ofstream outfile(output, std::ofstream::out | std::ofstream::binary);
    outfile.write((char*)serializedModel->data(), serializedModel->size());
}

inline void export_engine_sam_sample_encoder_and_mask_decoder(const std::string& f = "sam_onnx_example.onnx", const std::string& output = "sam_onnx_example.engine") {
    std::cout << "exporting " << f << " to " << output << std::endl;

    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    // auto parser = std::make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

#if SET_MAX_WORKSPACE
    // maximum workspace size
    const int workspace = MAX_WORKSPACE_IN_GB;  // GB
    config->setMaxWorkspaceSize(workspace * 1U << 30);
#endif
    config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMIN, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kOPT, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMAX, {1, 256, 64, 64 });

    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMIN, {3, 1, 2, 2});
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kOPT, {3, 1, 5, 2});
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMAX, {3, 1, 10, 2});

    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMIN, {2, 1, 2});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kOPT, {2, 1, 5});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMAX, {2, 1, 10});

    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMIN, { 1, 1, 256, 256});
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kOPT, { 1, 1, 256, 256 });
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMAX, { 1, 1, 256, 256 });

    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMIN, { 1,});
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kOPT, { 1,});
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMAX, { 1,});

    config->addOptimizationProfile(profile);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
    std::cout << "serializedModel->size(): " << serializedModel->size() << std::endl;

    std::cout << "saving model to " << output << std::endl;
    std::ofstream outfile(output, std::ofstream::out | std::ofstream::binary);
    outfile.write((char*)serializedModel->data(), serializedModel->size());
}

#if 0
inline void export_engine_sam_sample_encoder_and_mask_decoder_h(const std::string& f = "sam_onnx_example.onnx", const std::string& output = "sam_onnx_example.engine") {
    std::cout << "exporting " << f << " to " << output << std::endl;

    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    //auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    //auto parser = std::make_unique<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

#if SET_MAX_WORKSPACE
    // maximum workspace size
    const int workspace = MAX_WORKSPACE_IN_GB;  // GB
    config->setMaxWorkspaceSize(workspace * 1U << 30);
#endif 
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    config->setFlag(BuilderFlag::kFP16);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMIN, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kOPT, {1, 256, 64, 64 });
    // profile->setDimensions("image_embeddings", nvinfer1::OptProfileSelector::kMAX, {1, 256, 64, 64 });

    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMIN, { 3,1, 5,2 });
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kOPT, { 3,1, 5,2 });
    profile->setDimensions("point_coords", nvinfer1::OptProfileSelector::kMAX, { 3,1, 5,2 });

    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMIN, { 2,1, 5});
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kOPT, { 2,1, 5 });
    profile->setDimensions("point_labels", nvinfer1::OptProfileSelector::kMAX, { 2,1, 5 });

    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMIN, { 1, 1, 256, 256});
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kOPT, { 1, 1, 256, 256 });
    // profile->setDimensions("mask_input", nvinfer1::OptProfileSelector::kMAX, { 1, 1, 256, 256 });

    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMIN, { 1,1,});
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kOPT, { 1,1,});
    // profile->setDimensions("has_mask_input", nvinfer1::OptProfileSelector::kMAX, { 1,1,});

    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kMIN, {1,2,});
    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kOPT, {1,2,});
    profile->setDimensions("orig_im_size", nvinfer1::OptProfileSelector::kMAX, {1,2,});    

    config->addOptimizationProfile(profile);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
     std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
     std::cout << "serializedModel->size(): " << serializedModel->size() << std::endl;

     std::cout << "saving model to " << output << std::endl;
     std::ofstream outfile(output, std::ofstream::out | std::ofstream::binary);
     outfile.write((char*)serializedModel->data(), serializedModel->size());
}
#endif