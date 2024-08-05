#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

#include "macros.h"
#include "Files.h"
#include "TUM.h"
#include "FeatureManager.h"


using namespace cv;
using namespace std;
using namespace slamplay;

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag 

/***************************************************
* This example demonstrates how to train a dictionary 
  from the images available in the input dataset directory
*************************************************/

int main( int argc, char** argv )
{
    string dataset_dir = dataDir + "/loop_closure/";
    if( argc== 2) {
        dataset_dir = argv[1];
    } else {
      cout << "usage: " << argv[0] <<" <dataset dir>" << endl;
    }

    std::vector<std::string> filenames;
    getImageFilenames(dataset_dir, filenames);

    cout << "extracting features ... " << endl;
    const string feature_type = "orb";
    cout<<"type: " << feature_type <<endl;     
    Ptr<Feature2D> detector = getFeature2D(feature_type);
    int index = 1;
    vector<Mat> descriptors;
    for ( const string& file: filenames )
    {
        Mat image = imread(file);
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
        cout<<"extracting features from image " << index++ << " " << file << endl;
    }
    cout<<"extracted total "<<descriptors.size()*500<<" features."<<endl;
    
    // create vocabulary 
    cout<<"creating vocabulary, please wait ... "<<endl;
    DBoW3::Vocabulary vocab;
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "vocab.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}