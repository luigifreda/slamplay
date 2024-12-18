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

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace slamplay {

// for datasets from
// https://www.robots.ox.ac.uk/~mobile/IJRR_2008_Dataset/data.html
std::vector<std::string> readNewCollegeFilenames(const std::string& dir,
                                                 const bool skip_even = true) {
    std::vector<std::string> filenames;
    auto index_filename = dir + "/ImageCollectionCoordinates.txt";
    std::cout << "Opening index from " << index_filename << "\n";

    std::ifstream fin;
    fin.open(index_filename);

    if (!fin) {
        throw std::runtime_error("Failed to open index file");
    }

    while (!fin.eof()) {
        std::string l;
        std::getline(fin, l);

        if (!l.empty()) {
            std::stringstream ss;
            ss << l;
            std::string filename;
            ss >> filename;
            filenames.push_back(filename);
        }

        // Discard even-numbered images (for stereo datasets)
        if (skip_even) {
            std::getline(fin, l);
        }
    }

    return filenames;
}

}  // namespace slamplay