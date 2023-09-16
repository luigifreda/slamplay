#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//#include "extra.h" //use this if in OpenCV2

#include "macros.h"

using namespace std;
using namespace cv;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
string img1_file = dataDir + "/rgbd2/1.png";
string img2_file = dataDir + "/rgbd2/2.png";
//Camera internal reference, TUM Freiburg2
const Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

/****************************************************
*This program demonstrates how to use 2D-2D feature matching to estimate camera motion
***************************************************/

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  std::vector<KeyPoint> keypoints_1,
  std::vector<KeyPoint> keypoints_2,
  std::vector<DMatch> matches,
  Mat &R, Mat &t);

//Convert pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {

  if (argc == 3)
  {
    img1_file = argv[1];
    img2_file = argv[2];
  } 
  else if (argc != 3) 
  {
    cout << "usage: " << argv[0] <<" img1 img2" << endl;
  }

  //--read the image
  Mat img_1 = imread(img1_file, cv::IMREAD_COLOR);
  Mat img_2 = imread(img2_file, cv::IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "found a total " << matches.size() << " group matching point" << endl;

  //--estimate motion between two images
  Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  //--Verify that E=t^R*scale
  Mat t_x = (Mat_<double>(3, 3) 
    <<                  0, -t.at<double>(2, 0),  t.at<double>(1, 0),
       t.at<double>(2, 0),                   0, -t.at<double>(0, 0),
      -t.at<double>(1, 0),  t.at<double>(0, 0),                  0);

  cout << "t^R=" << endl << t_x * R << endl;

  //--verify the epipolar constraint
  for (DMatch m: matches) {
    Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
    Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    Mat d = y2.t() * t_x * R * y1;
    cout << "epipolar constraint = " << d << endl;
  }
  return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
  //--initialization
  Mat descriptors_1, descriptors_2;

  Ptr<FeatureDetector> detector = ORB::create();
  Ptr<DescriptorExtractor> descriptor = ORB::create();

  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
  
  //--Step 1: Detect Oriented FAST corner position
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //--Step 2: Calculate the BRIEF descriptor according to the corner position
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  //--Step 3: Match the BRIEF descriptors in the two images, using the Hamming distance
  vector<DMatch> match;
  //BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  //--Step 4: Match point pair screening
  double min_dist = 10000, max_dist = 0;

  //Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //When the distance between descriptors is greater than twice the minimum distance, it is considered that the match is wrong. But sometimes the minimum distance will be very small, set an experience value of 30 as the lower limit.
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {

  //--Convert the matching point to the form of vector<Point2f>
  vector<Point2f> points1;
  vector<Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  //--calculate the fundamental matrix
  Mat fundamental_matrix;
  fundamental_matrix = findFundamentalMat(points1, points2, cv::FM_8POINT);
  cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

  //--calculate the essential matrix
  // Point2d principal_point(325.1, 249.7);//Camera optical center, TUM dataset calibration value
  // double focal_length = 521;//Camera focal length, TUM dataset calibration value
  Mat essential_matrix;
  essential_matrix = findEssentialMat(points1, points2, K, cv::USAC_MAGSAC);
  cout << "essential_matrix is " << endl << essential_matrix << endl;

  //--calculate the homography matrix
  //--but in this example the scene is not flat, the homography matrix is ​​of little significance
  Mat homography_matrix;
  homography_matrix = findHomography(points1, points2, cv::USAC_MAGSAC, 3);
  cout << "homography_matrix is " << endl << homography_matrix << endl;

  //--Recover rotation and translation information from the essential matrix.
  //This function is only available in Opencv3
  recoverPose(essential_matrix, points1, points2, K, R, t);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;

}
