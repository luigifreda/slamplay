#include <stdio.h>
#include <iostream>

#include <memory>

#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#ifdef HAVE_OPENCV_CONTRIB
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif 


#include "macros.h"

using namespace std;
using namespace cv;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 

//string img1_file = dataDir + "/rgbd2/1.png";
//string img2_file = dataDir + "/rgbd2/2.png";
 
//string img1_file = dataDir + "/homography/box.png";
//string img2_file = dataDir + "/homography/box_in_scene.png";

string img1_file = dataDir + "/graf/img1.ppm";
string img2_file = dataDir + "/graf/img4.ppm";


enum DetectorType { 
    kDetectORB=0, 
    kDetectAKAZE=1, 
    kDetectBRISK=2, 
    kDetectSIFT=3, 
    kDetectSURF=4,
}; 
const std::string kDescriptorName[] = { 
    "ORB",      // 0
    "AKAZE",    // 1
    "BRISK",    // 2
    "SIFT",     // 3
    "SURF",     // 4
}; 

DetectorType detectorType = kDetectORB;
bool bUseFlannForMatching = false; 


// Camera internal reference, TUM Freiburg2
cv::Mat KTUM2 = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);


void readme()
{
    std::cout << " Usage: ./test <img1 train> <img2 query> maxDist descriptor_type" << std::endl;
}

void setCameraMatrixAndDist(cv::Mat& K, cv::Mat& distCoef) 
{
#if 0    
    const float fx = 667.3896484375;
    const float fy = 667.3896484375;
    const float cx = 672.755615234375;
    const float cy = 370.8421325683594;

    // 3x3 row-major matrix
    //     [fx  0 cx]
    // K = [ 0 fy cy]
    //     [ 0  0  1]        
    K = cv::Mat::zeros(3, 3, CV_32F);
    K.at<float>(0,0) = fx; K.at<float>(0,2) = cx;
    K.at<float>(1,1) = fy; K.at<float>(1,2) = cy;    
    K.at<float>(2,2) = 1.;

    // For "plumb_bob", D = [k1, k2, t1, t2, k3].        
    distCoef = cv::Mat::zeros(5,1,CV_32F);
#else
    K = KTUM2;
    distCoef = cv::Mat::zeros(5,1,CV_32F);    
#endif 
}

struct compare_descriptor_by_dist
{
    inline bool operator()(const cv::DMatch& a, const cv::DMatch& b)
    {
        return ( a.distance < b.distance);
    }
};

void matchesFilterMAD(std::vector<cv::DMatch>& matches, double &sigma_mad, const double max_dist)
{
    constexpr double factor = 1.5; // 1.5

    // estimate the NN's distance standard deviation
    double dist_median;
    sort(matches.begin(), matches.end(), compare_descriptor_by_dist());
    dist_median = matches[int(matches.size() / 2)].distance;
    sigma_mad = factor * 1.4826 * dist_median;
    
    /*for (size_t j = 0; j < matches_nn.size(); j++)
        matches_nn[j][0].distance = fabsf(matches_nn[j][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    sigma_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;*/
    
    const double th = std::min( (double)max_dist, (double)sigma_mad ); 
    std::cout << "matchesFilterMAD - th: " << th << std::endl;
        
    int i=matches.size()-1;
    for(; i>=0; i--)
    {
        std::cout << "distance[" <<i<<"]: " << matches[i].distance << std::endl; 
        if( matches[i].distance < th) break; 
    }
      
    if( i >=0)  
    {
        std::cout << "remaining " <<i+1<<" elements after mad" << std::endl;           
        matches.resize(i+1);
    }
    
}

void undistortImgPoints(const cv::Mat& K, const cv::Mat& distCoef, const vector<Point2f>& objectImgPoints, vector<Point2f>& objectImgPointsUn)
{
    int N = objectImgPoints.size();
        
    if(distCoef.at<float>(0)==0.0)
    {
        std::cout << "zero distortion coeff" << std::endl;
        objectImgPointsUn = objectImgPoints;
        return;
    }
    else
    {
        std::cout << "dist coeff: " << distCoef << std::endl; 
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=objectImgPoints[i].x;
        mat.at<float>(i,1)=objectImgPoints[i].y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,K,distCoef,cv::Mat(),K);
    mat=mat.reshape(1);

    // Fill undistorted point vector
    objectImgPointsUn.resize(N);
    for(int i=0; i<N; i++)
    {
        objectImgPointsUn[i].x=mat.at<float>(i,0);
        objectImgPointsUn[i].y=mat.at<float>(i,1);
    }
}

void setObjectPoints( const float scale, const cv::Size& imgSize, const cv::Mat& K, const cv::Mat& distCoef, const vector<Point2f>& objectImgPoints, vector<Point3f>& object3DPoints)
{
    vector<Point2f> objectImgPointsUn;
    std::cout << "undistorting points " << std::endl; 
    undistortImgPoints(K, distCoef, objectImgPoints, objectImgPointsUn);
    
    // 3x3 row-major matrix
    //     [fx  0 cx]
    // K = [ 0 fy cy]
    //     [ 0  0  1]         
    //const float fx = K.at<float>(0,0);
    //const float fy = K.at<float>(1,1);
    //const float cx = K.at<float>(0,2);
    //const float cy = K.at<float>(1,2);    
    
    const float cx = round(imgSize.width/2.);
    const float cy = round(imgSize.height/2.);        
    
    size_t N = objectImgPoints.size();
    object3DPoints.resize(N); 
    for(size_t jj=0; jj<N; jj++)
    {
        object3DPoints[jj].x =  (objectImgPoints[jj].x - cx)*scale;
        object3DPoints[jj].y = -(objectImgPoints[jj].y - cy)*scale;
        object3DPoints[jj].z = 0;
    }    
}

/** @function main */
int main(int argc, char** argv)
{
    if (argc == 3)
    {
        img1_file = argv[1];
        img2_file = argv[2];
    } 
    else if (argc < 3) 
    {
        readme();
    }
        
    double maxDescriptorDist = std::numeric_limits<double>::max();

    if(argc == 4) maxDescriptorDist = std::atoi( argv[3] );   
    if(argc == 5) detectorType = (DetectorType)std::atoi( argv[4] ); ;

    Mat img_train = imread(img1_file, cv::IMREAD_COLOR);
    Mat img_query = imread(img2_file, cv::IMREAD_COLOR);  
    
    std::cout << "max distance: " << maxDescriptorDist << std::endl; 

    if (!img_train.data || !img_query.data)
    {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    std::vector<KeyPoint> keypoints_train, keypoints_query;
    Mat descriptors_train, descriptors_query;

    const float nn_match_ratio = 0.9f;   // Nearest neighbor matching ratio        

    int normType=NORM_HAMMING;    
    
    const int nfeatures = 2000;
    const float scaleFactor = 1.2;
    const int nlevels = 8;

    cv::Ptr<Feature2D> detector;
    cv::Ptr<flann::IndexParams> pIndexParams;
    
    // parameters from https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    const int table_number = 12; 
    const int key_size = 20; 
    const int multi_probe_level = 2;    
    const int minHessian = 100;

    switch( detectorType ) 
    {

    case kDetectAKAZE:
        normType=NORM_HAMMING;                            
        detector = cv::AKAZE::create(); 
        pIndexParams = cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);
        break; 
        
    case kDetectBRISK:
        normType=NORM_HAMMING;
        detector = cv::BRISK::create();
        pIndexParams = cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);
        break; 
        
    case kDetectSIFT:
        normType=NORM_L2;
        detector = cv::SIFT::create(nfeatures, nlevels);   
        pIndexParams = cv::makePtr<flann::KDTreeIndexParams>();
        break; 

#ifdef HAVE_OPENCV_CONTRIB        
    case kDetectSURF:
        normType=NORM_L2;
        detector = cv::xfeatures2d::SURF::create(minHessian, nlevels);  
        pIndexParams = cv::makePtr<flann::KDTreeIndexParams>();        
        break;        
#endif 

    default: 
    case kDetectORB:
        normType=NORM_HAMMING;
        detector = cv::ORB::create(nfeatures, scaleFactor, nlevels);   
        pIndexParams = cv::makePtr<cv::flann::LshIndexParams>(table_number, key_size, multi_probe_level);
        break;         
        
    }
    
    std::cout << "using " << kDescriptorName[detectorType] << std::endl; 
    
    detector->detectAndCompute(img_train, noArray(), keypoints_train, descriptors_train);
    detector->detectAndCompute(img_query, noArray(), keypoints_query, descriptors_query);     
    
    
    std::cout << "features extracted ===========================================" << std::endl; 
    std::cout << "descriptors_train: " << descriptors_train.size() << std::endl; 
    std::cout << "descriptors_query: " << descriptors_query.size() << std::endl; 
    
    if( descriptors_train.empty() || descriptors_query.empty() )
    {        
        std::cout << "empty descriptors " << std::endl; 
        quick_exit(-1);        
    }
    
    std::vector< DMatch > matches;
    vector< vector<DMatch> > nn_matches;    
    
    std::unique_ptr<DescriptorMatcher> matcher; 
   
    // Matching descriptor vectors using FLANN or BF matcher    
    if(bUseFlannForMatching)
    {
        std::cout << "using flann matcher " << std::endl;
        matcher = std::make_unique<FlannBasedMatcher>(pIndexParams);
        //matcher->match(descriptors_query, descriptors_train, matches);
        matcher->knnMatch(descriptors_query /*query*/, descriptors_train/*train*/, nn_matches, 2);   
    }
    else
    {
        std::cout << "using BF matcher " << std::endl;    
        matcher = std::make_unique<BFMatcher>(normType);        
        matcher->knnMatch(descriptors_query /*query*/, descriptors_train/*train*/, nn_matches, 2);
    }
    
    std::cout << "nn matches: " << nn_matches.size() << std::endl; 
    //vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    //vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); i++)
    {
        //std::cout << "size nn_matches[" << i << "]: " << nn_matches[i].size() << std::endl;
        if(nn_matches[i].empty()) continue;
        
        const DMatch& first = nn_matches[i][0];
        if(nn_matches[i].size() == 2) 
        {
            const cv::DMatch& second = nn_matches[i][1];
            const float dist1 = first.distance;
            const int octave1 = keypoints_query[first.queryIdx].octave;
            const float dist2 = second.distance;
            const int octave2 = keypoints_query[second.queryIdx].octave;            
            if( (dist1 > nn_match_ratio * dist2) && ( octave1 == octave2 ) )
            {
                continue;
            }
        }
        //matched1.push_back(keypoints_object[first.queryIdx]);
        //matched2.push_back(keypoints_scene[first.trainIdx]);
        matches.push_back( first );
    }    
    
    
    std::cout << "features matched  ===========================================" << std::endl; 
    std::cout << "num matches: " << matches.size() << " (after distance ratio filter) " << std::endl; 

    double max_dist = 0.;
    double min_dist = std::numeric_limits<double>::max();

    //-- Quick calculation of max and min distances between keypoints
    for (size_t i = 0; i < matches.size(); i++)
    {
        double dist = matches[i].distance;
        //std::cout << "distance[" << i << "]: " << dist << std::endl; 
        if (dist < min_dist) min_dist = dist;  
        if (dist > max_dist) max_dist = dist;
    }

    printf("Max dist : %f \n", max_dist);
    printf("Min dist : %f \n", min_dist);
    
    double sigma_mad = 0;
    
#if 0   
    
    //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
    std::vector<DMatch> good_matches;
    const int thDist = std::min( 3 * min_dist, maxDescriptorDist) ;
  
    for (int i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < thDist)
        {
            good_matches.push_back(matches[i]);
        }
    }
#else
        
    std::vector<DMatch>& good_matches = matches;
    int total_matches = matches.size();
    
    matchesFilterMAD(good_matches, sigma_mad, maxDescriptorDist);
    printf("Sigma : %f \n", sigma_mad);    
    
#endif    
    
    std::cout << "good matches: " << good_matches.size() << ", perc: " <<  good_matches.size()*100./total_matches << "%, maxDist: " << maxDescriptorDist << std::endl;     
    

    std::cout << "computing homography   ======================================" << std::endl; 
    
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints_train[good_matches[i].trainIdx].pt);
        scene.push_back(keypoints_query[good_matches[i].queryIdx].pt);
    }
    
    double ransacReprojThreshold = 3; // default value 3
    std::vector<uchar> outMask;    
    cv::Mat H = cv::findHomography(obj, scene, cv::USAC_MAGSAC, ransacReprojThreshold, outMask);    
    //std::cout << "output mask " << outMask.type() <<", data: " <<  outMask << std::endl; 
    
    std::vector< DMatch > inlier_matches;
    vector<Point2f> objInliers;    
    vector<Point2f> sceneInliers;        
    for (size_t i = 0; i < good_matches.size(); i++)
    {
        //std::cout << "outMask[i]: " << (int)outMask[i] << std::endl;
#if 1        
        if( outMask[i]) // use the RANSAC outliers detection 
#else
        // take all the matches (just for testing )    
#endif
        {
            //inlier_matches.push_back(good_matches[i]);
            inlier_matches.push_back(DMatch(good_matches[i].trainIdx, good_matches[i].queryIdx, good_matches[i].distance));
            objInliers.push_back( keypoints_train[ good_matches[i].trainIdx ].pt );
            sceneInliers.push_back( keypoints_query[ good_matches[i].queryIdx ].pt );
        }
    }    
    
    std::cout << "inliers: " << inlier_matches.size() << std::endl; 
    

    std::cout << "computing reprojection error ================================" << std::endl;      
    std::vector<Point2f> obj_reproj;
    double reprojectionError = 0; 
    perspectiveTransform(objInliers, obj_reproj, H);    
    for(size_t ii=0; ii<sceneInliers.size(); ii++)
    {
        reprojectionError += pow2(sceneInliers[ii].x - obj_reproj[ii].x) + pow2(sceneInliers[ii].y - obj_reproj[ii].y);
    }
    reprojectionError = sqrt( reprojectionError/(2.*sceneInliers.size()) );
    std::cout << "reprojectionError: " << reprojectionError << std::endl;
    
#if 0
    std::cout << "pose estimation =============================================" << std::endl; 
    
    vector<Point3f> objectPoints; 
    double panelWidth = 1.5; // [m] corresponding to width 
    double scale = panelWidth/img_train.cols;
        
    cv::Mat K;
    cv::Mat distCoef;
    cv::Mat rvec;    
    cv::Mat tvec;

    std::cout << "set camera matrix" << std::endl;     
    setCameraMatrixAndDist(K, distCoef);
    std::cout << "set object points" << std::endl;     
    setObjectPoints(scale, img_train.size(), K, distCoef, objInliers, objectPoints);   
    std::cout << "solving pnp" << std::endl; 
    cv::solvePnP(objectPoints, sceneInliers, K, distCoef, rvec, tvec, /*useExtrinsicGuess*/ false, cv::SOLVEPNP_ITERATIVE);
    
    std::cout << "rvec: " << rvec << std::endl;
    cv::Mat matRot;
    cv::Rodrigues(rvec, matRot);
    std::cout << "matRot: " << matRot << std::endl;    
    std::cout << "tvec: " << tvec << std::endl;    
#endif 

    std::cout << "drawing =====================================================" << std::endl;     
    Mat img_matches;
    drawMatches(img_train, keypoints_train, img_query, keypoints_query, 
                inlier_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

   
    //-- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f(img_train.cols, 0);
    obj_corners[2] = cv::Point2f(img_train.cols, img_train.rows);
    obj_corners[3] = cv::Point2f(0, img_train.rows);
    std::vector<cv::Point2f> scene_corners(4);

    perspectiveTransform(obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line(img_matches, scene_corners[0] + Point2f(img_train.cols, 0), scene_corners[1] + Point2f(img_train.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f(img_train.cols, 0), scene_corners[2] + Point2f(img_train.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f(img_train.cols, 0), scene_corners[3] + Point2f(img_train.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f(img_train.cols, 0), scene_corners[0] + Point2f(img_train.cols, 0), Scalar(0, 255, 0), 4);
    

    //-- Show detected matches
    std::string nameWindow = "object detection";
    namedWindow( nameWindow, WINDOW_NORMAL );// Create a window for display.
    imshow(nameWindow, img_matches);

    waitKey(0);
    return 0;
}
