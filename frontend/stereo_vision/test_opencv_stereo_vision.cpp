#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>

#include <vector>
#include <string>
#include <Eigen/Core>

#include <pangolin/pangolin.h>
#include <unistd.h>
#include <future>

#include "macros.h"

using namespace std;
using namespace Eigen;

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 

#define COLOR 0
#if COLOR
string left_file = dataDir + "/stereo/kitti06-12-L-color.png";
string right_file = dataDir + "/stereo/kitti06-12-R-color.png";
#else
string left_file = dataDir + "/stereo/left.png";
string right_file = dataDir + "/stereo/right.png";
#endif 

using Vector6d = Eigen::Matrix<double,6,1>;
using Pointcloud6D = vector<Vector6d, Eigen::aligned_allocator<Vector6d>>; 

//Drawing in pangolin, already written, no adjustments needed
void showPointCloud(const Pointcloud6D &pointcloud);

int main(int argc, char **argv) {

#if COLOR
    //internal reference
    double fx = 707.0912, fy = 707.0912, cx = 601.8873, cy = 183.1104;
    //baseline
    double b = 0.53715;
#else 
    //internal reference
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    //baseline
    double b = 0.573;
#endif 

    //read image
    cv::Mat left = cv::imread(left_file, cv::IMREAD_COLOR);
    cv::Mat right = cv::imread(right_file, cv::IMREAD_COLOR);
#if 1
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32); // magic parameters
#else 
    const int nMaxDisparity = 96; 
    const int nWinSizeSBGM = 1; 
    const int dSmoothingFactor = 4; 
    const double dLambda = 8000.0;
    const double dSigma = 1.5; 
    const int nMaxXGrad = 25; 
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, nMaxDisparity, nWinSizeSBGM);
    sgbm->setP1(24 * nWinSizeSBGM * nWinSizeSBGM * dSmoothingFactor);
    sgbm->setP2(96 * nWinSizeSBGM * nWinSizeSBGM * dSmoothingFactor);
    sgbm->setPreFilterCap(nMaxXGrad);
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);
    auto wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
    auto right_matcher = cv::ximgproc::createRightMatcher(sgbm);  
    wls_filter->setLambda(dLambda);
    wls_filter->setSigmaColor(dSigma);  
#endif 
    cv::Mat disparity_sgbm, disparity;
    sgbm->compute(left, right, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);

    //generate point cloud
    Pointcloud6D pointcloud;
    Vector6d point;

    //If your machine is slow, please change the following v++ and u++ to v+=2, u+=2
    for (int v = 0; v < left.rows; v++)
        for (int u = 0; u < left.cols; u++) {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) continue;

            auto bgr = left.at<cv::Vec3b>(v, u); 
            point << 0, 0, 0, bgr[2]/ 255.0, bgr[1]/ 255.0, bgr[0]/ 255.0; //The first three dimensions are xyz, and the fourth dimension is color

            //Calculate the position of the point according to the binocular model
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }

    auto future = std::async(std::launch::async, [&](){
        cv::imshow("image", left );        
        cv::imshow("disparity", disparity / 96.0);
        cv::waitKey(0);
    });

    //draw point cloud
    showPointCloud(pointcloud);

    future.wait(); 

    return 0;
}

void showPointCloud(const Pointcloud6D &pointcloud) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[4], p[5]);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);//sleep 5 ms
    }
    return;
}