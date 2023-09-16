#include <boost/format.hpp>  //for formatting strings
#include <iostream>

int main(int argc, char **argv) 
{
  std::string color_file_prefix = "ciao/miao/bao/color";
  std::string depth_file_prefix = "ciao/miao/bao/depth";    
  boost::format fmt("%s/%d.%s"); //image file format
  int i = 1; 
  std::string color_file = (fmt % color_file_prefix % i % "png").str();
  std::string depth_file = (fmt % depth_file_prefix % i % "pgm").str(); 

return 0;
}