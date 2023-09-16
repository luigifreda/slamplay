#pragma once 

#include <iostream>
#include <filesystem>

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

// two different ways of getting file names 

inline void getFilenames(const std::string& directory, 
                         std::vector<std::string>& filenames) 
{
    namespace fs = std::filesystem;

    for (const auto & entry : fs::directory_iterator(directory)) {
        filenames.push_back(directory + entry.path().c_str());
    }
    std::sort(filenames.begin(), filenames.end());
}


inline void getImageFilenames(const std::string& directory, 
                              std::vector<std::string>& filenames) 
{
    using namespace boost::filesystem;

    filenames.clear();
    path dir(directory);

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".png" || ext == ".jpg" ||
            ext == ".ppm" || ext == ".jpeg") {
            filenames.push_back(it->string());
        }
    }
}


inline void getFilenamesWithType(const std::string& directory, 
                                 std::vector<std::string>& filenames,
                                 std::vector<std::string>& types) 
{
    using namespace boost::filesystem;

    filenames.clear();
    path dir(directory);

    // transform all the types in the list in lower case 
    for (auto it_type = types.begin(); it_type != types.end(); it_type++){
        std::string& type = *it_type;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    }

    // Retrieving, sorting and filtering filenames.
    std::vector<path> entries;
    copy(directory_iterator(dir), directory_iterator(), back_inserter(entries));
    sort(entries.begin(), entries.end());
    for (auto it = entries.begin(); it != entries.end(); it++) {
        std::string ext = it->extension().c_str();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        for (auto it_type = types.begin(); it_type != types.end(); it_type++){
            if (ext == *it_type) {
                filenames.push_back(it->string());
            }
        }
    }
}
