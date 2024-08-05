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