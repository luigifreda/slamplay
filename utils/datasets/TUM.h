
#pragma once 

#include <fstream>     
#include <iostream>
#include <vector>
#include <string>


bool readTUMFileNames(const std::string& dataset_dir, 
                   std::vector<std::string>& rgb_files, 
                   std::vector<std::string>& depth_files,
                   std::vector<double>& rgb_times, 
                   std::vector<double>& depth_times)
{
    std::ifstream fin ( dataset_dir+"/association.txt" );
    if ( !fin )
    {
        std::cout << "please generate the associate file called association.txt!" << std::endl;
        std::cout << "see https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools" << std::endl;
        return 1;
    }

    while ( !fin.eof() )
    {
        std::string rgb_time, rgb_file, depth_time, depth_file;
        fin>>rgb_time>>rgb_file>>depth_time>>depth_file;
        rgb_times.push_back ( atof ( rgb_time.c_str() ) );
        depth_times.push_back ( atof ( depth_time.c_str() ) );
        rgb_files.push_back ( dataset_dir+"/"+rgb_file );
        depth_files.push_back ( dataset_dir+"/"+depth_file );

        if ( fin.good() == false )
            break;
    }
    fin.close();
    return true; 
}