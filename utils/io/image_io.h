// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
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