#pragma once

#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "io/messages.h"

#include <boost/filesystem.hpp>
#include <opencv2/features2d.hpp>

namespace slamplay {

inline bool fileExists(const std::string& fileName) {
    std::ifstream infile(fileName.c_str());
    return infile.good();
}

inline bool pathExists(const std::string& path) {
    return (access(path.c_str(), F_OK) == 0);
}

// check if it is a file or a folder
inline bool isFile(const std::string& path) {
    if (!pathExists(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool createFolder(const std::string& folderPath) {
    if (mkdir(folderPath.c_str(), 0777) != 0) {
        if (errno == EEXIST) {
            std::cout << "Folder already exists!" << std::endl;
            return true;  // Folder already exists
        } else {
            std::cerr << "Failed to create folder! Error code: " << errno << std::endl;
            return false;  // Failed to create folder
        }
    }

    std::cout << "Folder created successfully!" << std::endl;
    return true;  // Folder created successfully
}

// inline bool fileExists(const std::string& file) {
//     struct stat file_status {};
//     if (stat(file.c_str(), &file_status) == 0 &&
//         (file_status.st_mode & S_IFREG)) {
//         return true;
//     }
//     return false;
// }

// two different ways of getting file names
inline void getFilenames(const std::string& directory,
                         std::vector<std::string>& filenames) {
    namespace fs = std::filesystem;

    for (const auto& entry : fs::directory_iterator(directory)) {
        // filenames.push_back(directory + entry.path().c_str());
        filenames.push_back(entry.path().c_str());
    }
    std::sort(filenames.begin(), filenames.end());
}

// inline bool getFilenames(const std::string& path, std::vector<std::string>& filenames) {
//     DIR* pDir;
//     struct dirent* ptr;
//     if (!(pDir = opendir(path.c_str()))) {
//         std::cerr << "Current folder doesn't exist!" << std::endl;
//         return false;
//     }
//     while ((ptr = readdir(pDir)) != nullptr) {
//         if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
//             filenames.push_back(path + "/" + ptr->d_name);
//         }
//     }
//     closedir(pDir);
//     std::sort(filenames.begin(), filenames.end());
//     return true;
// }

inline void getImageFilenames(const std::string& directory,
                              std::vector<std::string>& filenames) {
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
                                 std::vector<std::string>& types) {
    using namespace boost::filesystem;

    filenames.clear();
    path dir(directory);

    // transform all the types in the list in lower case
    for (auto it_type = types.begin(); it_type != types.end(); it_type++) {
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

        for (auto it_type = types.begin(); it_type != types.end(); it_type++) {
            if (ext == *it_type) {
                filenames.push_back(it->string());
            }
        }
    }
}

inline bool hasFileSuffix(const std::string& str, const std::string& suffix) {
    std::size_t index = str.find(suffix, str.size() - suffix.size());
    return (index != std::string::npos);
}

inline std::string getFileNameWithouExtension(const std::string& fileName) {
    std::size_t pos = fileName.rfind('.');
    if (pos == std::string::npos) return fileName;
    std::string resString(fileName.substr(0, pos));
    return resString;
}

inline std::vector<std::string> readNameListFromFile(const std::string filename) {
    std::vector<std::string> names;
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        MSG_ASSERT(false, "Cannot open file: " + filename);
        return names;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        names.emplace_back(line);
    }
    return names;
}

inline void concatenateFolderAndFileNameBase(
    const std::string& folder, const std::string& file_name,
    std::string* path) {
    *path = folder;
    if (path->back() != '/') {
        *path += '/';
    }
    *path = *path + file_name;
}

inline std::string concatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name) {
    std::string path;
    concatenateFolderAndFileNameBase(folder, file_name, &path);
    return path;
}

}  // namespace slamplay