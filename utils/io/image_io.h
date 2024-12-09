#pragma once

#include <dirent.h>
#include <filesystem>
#include <iostream>
#include <vector>

#include "io/string_utils.h"

namespace slamplay {

std::vector<std::string> GetPngFiles(const std::string& strPngDir) {
    std::cout << "Reading png files from " << strPngDir << std::endl;
    struct dirent** namelist;
    std::vector<std::string> ret;
    int n = scandir(strPngDir.c_str(), &namelist, [](const struct dirent* cur) -> int {
        std::string str(cur->d_name);
        return str.find(".png") != std::string::npos; }, alphasort);

    if (n < 0) {
        return ret;
    }

    for (int i = 0; i < n; i++) {
        std::string filepath(namelist[i]->d_name);
        ret.push_back("/" + filepath);
    }

    free(namelist);
    return ret;
}

std::vector<std::string> GetImageFiles(const std::string& strImageDir, const std::string& typeName = ".png") {
    std::cout << "Reading image files from " << strImageDir << std::endl;
    std::vector<std::string> ret;

    for (const auto& entry : std::filesystem::directory_iterator(strImageDir)) {
        const std::string& filename = entry.path().filename().string();
// check if c++20
#if __cplusplus >= 202002L
        if (filename.ends_with(typeName)) {
#else
        if (hasEnding(filename, typeName)) {
#endif
            ret.push_back(entry.path().string());
        }
    }

    return ret;
}

}  // namespace slamplay