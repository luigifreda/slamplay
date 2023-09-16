#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

#include <macros.h>

using namespace std;
using namespace cv;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
string img1_file = dataDir + "/rgbd2/1.png";
string img2_file = dataDir + "/rgbd2/2.png";
string depth1_file = dataDir + "/rgbd2/1_depth.png";
string depth2_file = dataDir + "/rgbd2/2_depth.png";
//Camera internal reference, TUM Freiburg2
const Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

//Convert pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t
);

void poseOptimization(
  const vector<Point3f> &points_3d,
  const vector<Point3f> &points_2d,
  Mat &R, Mat &t
);


double computeAlignmentError(const vector<Point3f>& pts1, 
                              const vector<Point3f>& pts2,
                              const cv::Mat& R,
                              const cv::Mat& t);

///vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual void setToOriginImpl() override {
    _estimate = Sophus::SE3d();
  }

  ///left multiplication on SE3
  virtual void oplusImpl(const double *update) override {
    Eigen::Matrix<double, 6, 1> update_eigen;
    update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
    _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
  }

  virtual bool read(istream &in) override {}

  virtual bool write(ostream &out) const override {}
};

///g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

  virtual void computeError() override {
    const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
    _error = _measurement - pose->estimate() * _point;
  }

  virtual void linearizeOplus() override {
    VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
    Sophus::SE3d T = pose->estimate();
    Eigen::Vector3d xyz_trans = T * _point;
    _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
  }

  bool read(istream &in) {}

  bool write(ostream &out) const {}

protected:
  Eigen::Vector3d _point;
};

int main(int argc, char **argv) 
{
  if (argc == 5)
  {
    img1_file = argv[1];
    img2_file = argv[2];
    depth1_file = argv[3];
    depth2_file = argv[4];    
  } 
  else if (argc != 5) 
  {
    cout << "usage: " << argv[0] <<" img1 img2 depth1 depth2" << endl;
  }
  //--read the image
  Mat img_1 = imread(img1_file, cv::IMREAD_COLOR);
  Mat img_2 = imread(img2_file, cv::IMREAD_COLOR);
  assert(img_1.data && img_2.data && "Can not load images!");

  vector<KeyPoint> keypoints_1, keypoints_2;
  vector<DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "found a total " << matches.size() << " group matching points" << endl;  

  //Create 3D points
  Mat depth1 = imread(depth1_file, cv::IMREAD_UNCHANGED);//The depth map is a 16-bit unsigned number, a single-channel image
  Mat depth2 = imread(depth2_file, cv::IMREAD_UNCHANGED);//The depth map is a 16-bit unsigned number, a single-channel image
  vector<Point3f> pts1, pts2;

  for (DMatch m:matches) 
  {
    ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
    if (d1 == 0 || d2 == 0)//bad depth
      continue;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    float dd1 = float(d1) / 5000.0;
    float dd2 = float(d2) / 5000.0;
    pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
    pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
  }

  cout << "3d-3d pairs: " << pts1.size() << endl;
  Mat R, t;

  cout << "\nStarting SVD-base registration" << endl;

  pose_estimation_3d3d(pts1, pts2, R, t);
  cout << "Registration via SVD results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;

  double error1 = computeAlignmentError(pts1, pts2, R, t);
  cout << "average alignment error: " << error1 << std::endl; 

  cout << "\nStarting pose optimization" << endl;

  poseOptimization(pts1, pts2, R, t);
  cout << "Pose optimization results: " << endl;
  cout << "R = " << R << endl;
  cout << "t = " << t << endl;
  cout << "R_inv = " << R.t() << endl;
  cout << "t_inv = " << -R.t() * t << endl;  

  double error2 = computeAlignmentError(pts1, pts2, R, t);
  cout << "average alignment error: " << error2 << std::endl; 

  std::cout << "\nthe end\n" << std::endl; 

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
  return Point2d(
    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
  );
}

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                          const vector<Point3f> &pts2,
                          Mat &R, Mat &t) {
  Point3f p1, p2;//center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = Point3f(Vec3f(p1) / N);
  p2 = Point3f(Vec3f(p2) / N);
  vector<Point3f> q1(N), q2(N);//remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  //compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W =" << W << endl;

  //SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U =" << U << endl;
  cout << "V =" << V << endl;

  Eigen::Matrix3d R_ = U * V.transpose();
  if (R_.determinant() < 0) 
  {
    //R_ = -R_;
    R_ = U  * Eigen::DiagonalMatrix<double,3>(1,1,-1) * V.transpose(); // we must guarantee det(R) = 1 > 0
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

//convert to cv::Mat
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void poseOptimization(
  const vector<Point3f> &pts1,
  const vector<Point3f> &pts2,
  Mat &R, Mat &t) 
{
  //Build graph optimization, first set g2o
  typedef g2o::BlockSolverX BlockSolverType;
  typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;//linear solver type
  //Gradient descent method, you can choose from GN, LM, DogLeg
  auto solver = new g2o::OptimizationAlgorithmLevenberg(
    g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  g2o::SparseOptimizer optimizer;//graph model
  optimizer.setAlgorithm(solver);//set up the solver
  optimizer.setVerbose(true);//turn on debug output

  //vertex
  VertexPose *pose = new VertexPose();//camera pose
  pose->setId(0);
  pose->setEstimate(Sophus::SE3d());
  optimizer.addVertex(pose);

  //edges
  for (size_t i = 0; i < pts1.size(); i++) 
  {
    EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
    edge->setVertex(0, pose);
    edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
    edge->setInformation(Eigen::Matrix3d::Identity());
    optimizer.addEdge(edge);
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  optimizer.initializeOptimization();
  optimizer.optimize(10);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

  cout << endl << "after optimization:" << endl;
  cout << "T=\n" << pose->estimate().matrix() << endl;

  //convert to cv::Mat
  Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
  Eigen::Vector3d t_ = pose->estimate().translation();
  R = (Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
  cout << endl << "end pose optimization" << endl;  
}


double computeAlignmentError(const vector<Point3f>& pts1, 
                              const vector<Point3f>& pts2,
                              const cv::Mat& R,
                              const cv::Mat& t)
{
  double avg_error = 0;
  Mat_<double> p2i_aligned(3, 1);
  Mat_<double> ei(3, 1);
  for (size_t i = 0; i < pts1.size(); i++) 
  {
    p2i_aligned = R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t;
    // compute the error ei = p1 - R *p2 + t
    ei = (Mat_<double>(3, 1) << pts1[i].x, pts1[i].y, pts1[i].z) - p2i_aligned;
    avg_error += ei.dot(ei); 
  }
  avg_error = sqrt(avg_error/(2*pts1.size())); 
  return avg_error; 
};