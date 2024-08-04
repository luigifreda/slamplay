
#include "extractors/HFNetSettings.h"

#include <iostream>

using namespace std;

namespace hfnet {

const std::string HFNetSettings::kDataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
const std::string HFNetSettings::kHFNetConfigFile = kDataDir + "/hfnet/config.yaml";

HFNetSettings::HFNetSettings(const std::string& configFile) {
    readConfig(configFile);
}

bool HFNetSettings::readConfig(const std::string& configFile) {
    auto node = YAML::LoadFile(configFile.c_str());
    if (!node) {
        MSG_ERROR("Didn't find config file. Exiting.");
        return 0;
    }

    const auto hfnetNode = node["HFNet"];

    std::string type = getParam(hfnetNode, "Extractor:type", std::string("HFNetTF"));
    if (type == "HFNetTF") {
        modelType_ = kHFNetTFModel;
    } else if (type == "HFNetRT") {
        modelType_ = kHFNetRTModel;
    } else if (type == "HFNetVINO") {
        modelType_ = kHFNetVINOModel;
    } else {
        MSG_ERROR("Wrong extractor type in setting file!");
        exit(-1);
    }

    scaleFactor_ = getParam(hfnetNode, "Extractor:scaleFactor", scaleFactor_);
    nLevels_ = getParam(hfnetNode, "Extractor:nLevels", nLevels_);
    nFeatures_ = getParam(hfnetNode, "Extractor:nFeatures", nFeatures_);
    threshold_ = getParam(hfnetNode, "Extractor:threshold", threshold_);
    strModelPath_ = kDataDir + "/" + getParam(hfnetNode, "Extractor:modelPath", strModelPath_);
    imageSize_.width = getParam(hfnetNode, "Extractor:imageSize:width", imageSize_.width);
    imageSize_.height = getParam(hfnetNode, "Extractor:imageSize:height", imageSize_.height);

    std::string datasetBasePath = getParam(hfnetNode, "Dataset:base", std::string());
    std::string datasetSequenceName = getParam(hfnetNode, "Dataset:sequence", std::string());
    datasetPath_ = datasetBasePath + "/" + datasetSequenceName;
    MSG_INFO("Dataset path: " << datasetPath_);
    if (datasetBasePath.empty() && datasetSequenceName.empty()) {
        MSG_ERROR("The dataset base path or the dataset sequence name is empty. Exiting.");
        exit(-1);
    }

    if (modelType_ == kHFNetVINOModel) {
        scaleFactor_ = 1.0;
        nLevels_ = 1;
        cout << "Because the HFNetVINO model is too time-consuming, the image pyremid function is disabled." << endl;
    }

    return true;
}

ostream& operator<<(std::ostream& output, const HFNetSettings& settings) {
    output << "Hnet settings: " << endl;
    output << "\t-Model type: " << gStrModelTypeName[settings.modelType_] << endl;
    output << "\t-Scale factor of image pyramid: " << settings.scaleFactor_ << endl;
    output << "\t-Levels of image pyramid: " << settings.nLevels_ << endl;
    output << "\t-Features per image: " << settings.nFeatures_ << endl;
    output << "\t-Detector threshold: " << settings.threshold_ << endl;
    output << "\t-Load model path: " << settings.strModelPath_ << endl;

    return output;
}

}  // namespace hfnet
