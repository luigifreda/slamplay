#include <math.h>
#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "macros.h"
#include "yamlUtils.h"

std::string dataDir = STR(DATA_DIR);  // DATA_DIR set by compilers flag
std::string configFile = dataDir + "/config.yaml";

struct YamlConfig {
    std::string inputFilePath;
    struct SubParameters {
        double param1 = 0.0;
        std::string param2;
        bool param3 = false;
    } subParameters;
};

bool readYamlConfig(const YAML::Node& node, YamlConfig& config) {
    MSG_INFO_STREAM("Reading yaml configuration");
    config.inputFilePath = getParam(node, std::string("input_file_path"), std::string());
    config.subParameters.param1 = getParam(node, "subparameters:param1", config.subParameters.param1);
    config.subParameters.param2 = getParam(node, "subparameters:param2", config.subParameters.param2);
    config.subParameters.param3 = getParam(node, "subparameters:param3", config.subParameters.param3);
    return true;
}

int main(int argc, char** argv) {
    std::cout << "reading " << configFile << std::endl;
    auto node = YAML::LoadFile(configFile);
    if (!node) {
        MSG_ERROR("Didn't find config file. Exiting.");
        return 0;
    }

    const auto nodeTest = node["Test"];
    if (!nodeTest) {
        MSG_ERROR("Didn't find Test node in config file. Exiting.");
        return 0;
    }

    MSG_INFO_STREAM("Reading yaml configuration");
    YamlConfig config;
    if (!readYamlConfig(nodeTest, config)) {
        MSG_ERROR("Could not read yaml config. Exiting.");
        return 0;
    }
    return 0;
}
