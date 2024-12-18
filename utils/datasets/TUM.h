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

bool readTUMFileNames(const std::string& dataset_dir,
                      std::vector<std::string>& rgb_files,
                      std::vector<std::string>& depth_files,
                      std::vector<double>& rgb_times,
                      std::vector<double>& depth_times) {
    std::ifstream fin(dataset_dir + "/association.txt");
    if (!fin)
    {
        std::cout << "please generate the associate file called association.txt!" << std::endl;
        std::cout << "see https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools" << std::endl;
        return 1;
    }

    while (!fin.eof())
    {
        std::string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if (fin.good() == false)
            break;
    }
    fin.close();
    return true;
}

}