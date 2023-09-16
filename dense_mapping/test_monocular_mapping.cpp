#include <iostream>
#include <vector>
#include <fstream>

//#include <boost/timer.hpp>

//for dormitories
#include <sophus/se3.hpp>

//for own
#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "macros.h"
#include "cam_utils.h"
#include "image_error.h"
#include "PointCloudViz.h"
#include "pointcloud_from_image_depth.h"

using Sophus::SE3d;
using namespace std;
using namespace cv;

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag 

#define ENABLE_VIZ 0

#define USE_INVERSE_DEPTH_FOR_FILTERING 0

/**********************************************
* This program demonstrates the dense depth estimation of a monocular camera under a known trajectory
* Using epipolar search + NCC matching, corresponding to Section 12.2 of the book
* Please note that this program is not perfect, you can definitely improve it -I'm actually exposing some problems on purpose (this is an excuse).
***********************************************/
//------------------------------------------------------------------
//parameters
const int border = 20;//edge width
const int width = 640;//image width
const int height = 480;//image height
const double fx = 481.2f;//internal parameters of the camera
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3;//The half-width of the window taken by NCC
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);//NCC window area
#if USE_INVERSE_DEPTH_FOR_FILTERING  
const double min_cov = 0.0001; //Convergence determination: minimum variance
const double max_cov = 1; //Divergence judgment: maximum variance
#else 
const double good_error = 0.01; 
const double min_cov = good_error*good_error; //Convergence determination: minimum variance
const double max_cov = 10; //Divergence judgment: maximum variance
#endif 
const double good_cov = 2*min_cov; // (factor*sigma)^2  // just used for visualization 
//------------------------------------------------------------------
//important functions
///Read data from the REMODE dataset
bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    vector<SE3d> &poses,
    cv::Mat &ref_depth
);

/**
*Update the depth estimate based on the new image
*@param ref reference image
*@param curr current image
*@param T_C_R The pose from the reference image to the current image
*@param depth depth
*@param depth_cov depth variance
*/
void update(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    Mat &depth,
    Mat &depth_cov2
);

/**
*Polar search
*@param ref reference image
*@param curr current image
*@param T_C_R pose
*@param pt_ref the position of the point in the reference image
*@param depth_mu depth mean
*@param depth_cov depth variance
*@param pt_curr current point
*@param epipolar_direction epipolar direction
*@return success
*/
bool epipolarSearch(
    const Mat &ref,
    const Mat &curr,
    const SE3d &T_C_R,
    const Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Vector2d &pt_curr,
    Vector2d &epipolar_direction
);

/**
*Update depth filter
*@param pt_ref reference image point
*@param pt_curr current image point
*@param T_C_R pose
*@param epipolar_direction epipolar direction
*@param depth mean depth
*@param depth_cov2 depth direction
*@return success
*/
bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2
);

/**
*Calculate NCC score
*@param ref reference image
*@param curr current image
*@param pt_ref reference point
*@param pt_curr current point
*@return NCC score
*/
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

//Bilinear grayscale interpolation
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
    uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return ((1 - xx) * (1 - yy) * double(d[0]) +
            xx * (1 - yy) * double(d[1]) +
            (1 - xx) * yy * double(d[img.step]) +
            xx * yy * double(d[img.step + 1])) / 255.0;
}

//------------------------------------------------------------------
//some gadgets
//Show estimated depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate, const Mat &depth_variance, const double factor=0.4) {
    imshow("depth_truth", depth_truth * factor);
    imshow("depth_estimate", depth_estimate * factor);

    cv::Rect roi(border, border, width-2*border, height-2*border); 
    cv::Mat depth_truth_roi = cv::Mat(depth_truth,roi);
    cv::Mat depth_estimate_roi = cv::Mat(depth_estimate,roi);    
    cv::Mat depth_error_roi = depth_truth_roi - depth_estimate_roi;
    imshow("depth_error", depth_error_roi  * factor);

    imshow("depth_variance", depth_variance * factor);        

#if 0
    plotImageErrorWithColorbar(depth_error_roi);
#endif 

    waitKey(1);
}

// Get a mask with all pixels that have variance < max_variance
cv::Mat getMaskFromVariance(const cv::Mat& variance, const double max_variance)
{
    cv::Mat mask;
    cv::threshold(variance, mask, max_variance, 255, cv::THRESH_BINARY_INV); 
    mask.convertTo(mask, CV_8U);
    return mask; 
}

//Pixel to camera coordinate system
inline Vector3d px2cam(const Vector2d& px) {
    return Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}

//Camera coordinate system to pixel
inline Vector2d cam2px(const Vector3d& p_cam) {
    return Vector2d(
        p_cam(0, 0) * fx / p_cam(2, 0) + cx,
        p_cam(1, 0) * fy / p_cam(2, 0) + cy
    );
}

//Check if a point is inside the image bounding box
inline bool inside(const Vector2d &pt) {
    return pt(0, 0) >= border && pt(1, 0) >= border
           && pt(0, 0) + border < width && pt(1, 0) + border <= height;
}

// Show Polar Match
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// Show polar lines
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr);

/// Evaluation Depth Estimation
void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate, const Mat &depth_variance, const double max_variance);
//------------------------------------------------------------------

// Define the format used by the point cloud: XYZRGB is used here
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int main(int argc, char **argv) 
{
    string dataset_dir = dataDir + "/remode_test_data/test_data/";
    if( argc== 2) {
        dataset_dir = argv[1];
    } else {
      cout << "usage: " << argv[0] <<" <dataset dir>" << endl;
    }

    PointCloudViz<PointCloud> viz;
    viz.start();  

    // read data from dataset
    vector<string> color_image_files;
    vector<SE3d> poses_TWC;
    Mat ref_depth;
    bool ret = readDatasetFiles(dataset_dir, color_image_files, poses_TWC, ref_depth);
    if (ret == false) {
        cout << "Reading image files failed!" << endl;
        return -1;
    }
    cout << "read total " << color_image_files.size() << " files." << endl;

    //first image
    Mat ref = imread(color_image_files[0], cv::IMREAD_GRAYSCALE); // gray-scale image
    MSG_ASSERT((ref.cols==width) && (ref.rows==height),"Shoud be equal to the one set above!"); 

    Mat ref_color = imread(color_image_files[0], cv::IMREAD_COLOR); // for visualization  

    SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0; // Depth initial value
#if USE_INVERSE_DEPTH_FOR_FILTERING    
    double init_cov2 = 0.5; // Initial value of variance
#else 
    double init_cov2 = 3.0; // Initial value of variance
#endif     
    MSG_ASSERT(init_cov2<max_cov,"Please increase max_cov above the init cov");
    Mat depth(height, width, CV_64F, init_depth); // depth map
    Mat depth_cov2(height, width, CV_64F, init_cov2); // depth map variance; 
                                                      // in the case inverse depth parametrization, this represent the variance matrix of the inverse depth

    PointCloud::Ptr pointcloud(new PointCloud); 
    const Intrinsics intrinsics{fx,fy,cx,cy};
    const Eigen::Isometry3d Twc = Eigen::Isometry3d::Identity(); 

    for (size_t index = 1; index < color_image_files.size(); index++) {     
        cout << "*** loop " << index << " ***" << endl;
        Mat curr = imread(color_image_files[index], 0);
        if (curr.data == nullptr) continue;
        SE3d pose_curr_TWC = poses_TWC[index];
        SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // Coordinate transformation relationship: T_C_W *T_W_R = T_C_R
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaludateDepth(ref_depth, depth, depth_cov2, good_cov);
        plotDepth(ref_depth, depth, depth_cov2);


        // get a mask for viz: we just want to visualize the points where the variance is smaller than a threshold 
        cv::Mat mask = getMaskFromVariance(depth_cov2, good_cov); 
        imshow("mask", mask); 

        // here, depth represents the distance |OP| from camera center to 3D point P 
        getPointCloudFromImageAndDistance(ref_color, depth, mask, intrinsics, border, Twc, *pointcloud);        
        viz.update(*pointcloud); 

        imshow("image", curr);
        waitKey(1);
    }

    cout << "estimation returns, saving depth map ..." << endl;
    imwrite("depth.png", depth);
    cout << "done." << endl;

    cout << "press a key on cv window to exit ..." << endl; 
    waitKey(0);

    return 0;
}

bool readDatasetFiles(
    const string &path,
    vector<string> &color_image_files,
    std::vector<SE3d> &poses,
    cv::Mat &ref_depth) 
{
    ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
    if (!fin) return false;

    while (!fin.eof()) {
        // Data format: image file name tx, ty, tz, qx, qy, qz, qw, note that it is TWC instead of TCW
        string image;
        fin >> image;
        double data[7];
        for (double &d:data) fin >> d;

        color_image_files.push_back(path + string("/images/") + image);
        poses.push_back(
            SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                 Vector3d(data[0], data[1], data[2]))
        );
        if (!fin.good()) break;
    }
    fin.close();

    // load reference depth
    fin.open(path + "/depthmaps/scene_000.depth");
    ref_depth = cv::Mat(height, width, CV_64F);
    if (!fin) return false;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            double depth = 0;
            fin >> depth;
            ref_depth.ptr<double>(y)[x] = depth / 100.0;
        }

    return true;
}

//Update the entire depth map
void update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) 
{
    #pragma omp parallel for 
    for (int y = border; y < height - border; y++) 
    {
        const double* depth_ptr_y = depth.ptr<double>(y);         
        const double* depth_cov2_ptr_y = depth_cov2.ptr<double>(y); 
        
        #pragma omp parallel for         
        for (int x = border; x < width - border; x++)            
        {
            //loop through each pixel
            if (depth_cov2_ptr_y[x] < min_cov || depth_cov2_ptr_y[x] > max_cov) // Depth has converged or diverged
                continue;
            //search for a match of (x,y) on the epipolar line
            Vector2d pt_curr;
            Vector2d epipolar_direction;
            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Vector2d(x, y),
                depth_ptr_y[x],
                sqrt(depth_cov2_ptr_y[x]),
                pt_curr,
                epipolar_direction
            );

            if (ret == false)//match failed
                continue;

#if ENABLE_VIZ
            // Uncomment this to show matches
            showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);
#endif             

            // The match is successful, update the depth map
            updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
        }
    }
}

// polar line search
// For the method, see the two sections of 12.2 and 12.3 in the book
bool epipolarSearch(
    const Mat &ref, const Mat &curr,
    const SE3d &T_C_R, const Vector2d &pt_ref,
    const double &depth_mu, const double &depth_cov,
    Vector2d &pt_curr, Vector2d &epipolar_direction) 
{
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d P_ref = f_ref * depth_mu;//P vector of reference frame

    Vector2d px_mean_curr = cam2px(T_C_R * P_ref);//Pixels projected by depth mean
#if USE_INVERSE_DEPTH_FOR_FILTERING    
    const double inv_d_mu = 1.0/depth_mu; 
    const double inv_d_min = inv_d_mu - 3 * depth_cov, inv_d_max = inv_d_mu + 3 * depth_cov;
    double d_min = 1.0/inv_d_max, d_max = 1.0/inv_d_min;
#else
    double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
#endif     
    if (d_min < 0.1) d_min = 0.1;
    Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));//Pixels projected by minimum depth
    Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max));//Pixels projected by maximum depth

    Vector2d epipolar_line = px_max_curr - px_min_curr;//polar line (line segment form)
    epipolar_direction = epipolar_line;//polar direction
    epipolar_direction.normalize();
    double half_length = 0.5 * epipolar_line.norm();//Half length of epipolar line segment
    if (half_length > 100) half_length = 100;//we don't want to search for too many things

#if ENABLE_VIZ
    // Uncomment this sentence to display epipolar lines (line segments)
    showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );
#endif 

    // Search on the epipolar line, take the depth mean point as the center, and take half lengths on the left and right sides
    double best_ncc = -1.0;
    Vector2d best_px_curr;
    for (double l = -half_length; l <= half_length; l += 0.7) {//l+=sqrt(2)/2
        Vector2d px_curr = px_mean_curr + l * epipolar_direction;//Waiting points
        if (!inside(px_curr))
            continue;
        //Calculate the NCC between the point to be matched and the reference frame
        double ncc = NCC(ref, curr, pt_ref, px_curr);
        if (ncc > best_ncc) {
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }
    if (best_ncc < 0.85f)//only trust matches with high NCC
        return false;
    pt_curr = best_px_curr;
    return true;
}

double NCC(const Mat &ref, const Mat &curr,
           const Vector2d &pt_ref, const Vector2d &pt_curr) 
{
    //zero mean -normalized cross correlation
    //Calculate the mean first
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr;//The mean of the reference frame and the current frame
    for (int y = -ncc_window_size; y <= ncc_window_size; y++)    
        for (int x = -ncc_window_size; x <= ncc_window_size; x++)
        {
            double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
            mean_ref += value_ref;

            double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
            mean_curr += value_curr;

            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;

    //calculation Zero mean NCC
    double numerator = 0, denominator1 = 0, denominator2 = 0;
    for (size_t i = 0; i < values_ref.size(); i++) {
        double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
        numerator += n;
        denominator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
        denominator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
    }
    return numerator / sqrt(denominator1 * denominator2 + 1e-10);//prevent denominator from appearing zero
}

bool updateDepthFilter(
    const Vector2d &pt_ref,
    const Vector2d &pt_curr,
    const SE3d &T_C_R,
    const Vector2d &epipolar_direction,
    Mat &depth,
    Mat &depth_cov2) 
{
    //I don't know if anyone still reads this paragraph
    //calculate depth with triangulation
    SE3d T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    Vector3d f_curr = px2cam(pt_curr);
    f_curr.normalize();

    //equation
    //d_ref *f_ref = d_cur *( R_RC *f_cur ) + t_RC
    //f2 = R_RC *f_cur
    //Transform into the following matrix equations
    //=> [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref]   [f_ref^T t]
    //   [   f_2^T f_ref,    -f2^T f2 ] [d_cur] = [f2^T t   ]
    const Vector3d t = T_R_C.translation();
    const Vector3d f2 = T_R_C.so3() * f_curr;
    const Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
    Matrix2d A;
    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);
#if 0    
    const Vector2d ans = A.inverse() * b;
#else 
    ColPivHouseholderQR<Matrix2d> dec(A);
    const Vector2d ans = dec.solve(b);
#endif 
    const Vector3d xm = ans[0] * f_ref;//the result on the ref side
    const Vector3d xn = t + ans[1] * f2;//cur result
    const Vector3d p_esti = (xm + xn) / 2.0;//The position of P, take the average of the two
    const double depth_estimation = p_esti.norm();//depth value

    //calculate uncertainty (in one pixel as error)
    //Vector3d p = f_ref * depth_estimation;
    //Vector3d a = p - t;
    const double t_norm = t.norm();
    //double a_norm = a.norm();
    const double alpha = acos(f_ref.dot(t) / t_norm);
    //double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
    f_curr_prime.normalize();
    const double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    const double gamma = M_PI - alpha - beta_prime;
    const double p_prime_norm = t_norm * sin(beta_prime) / sin(gamma);

#if USE_INVERSE_DEPTH_FOR_FILTERING    
    const double d_cov = 1.0/p_prime_norm - 1.0/depth_estimation;
#else 
    const double d_cov = p_prime_norm - depth_estimation;
#endif     
    const double d_cov2 = d_cov * d_cov;

    //Gaussian fusion
#if USE_INVERSE_DEPTH_FOR_FILTERING
    const double mu = 1.0/depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
#else 
    const double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
#endif     

    const double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

#if USE_INVERSE_DEPTH_FOR_FILTERING
    const double mu_fuse = (d_cov2 * mu + sigma2 * 1.0/depth_estimation) / (sigma2 + d_cov2 + 1e-10); //prevent denominator from appearing zero
#else 
    const double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2 + 1e-10); //prevent denominator from appearing zero
#endif     

    const double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2 + 1e-10); //prevent denominator from appearing zero

#if USE_INVERSE_DEPTH_FOR_FILTERING
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = 1.0/mu_fuse;
#else 
    depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;    
#endif     
    depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}



void evaludateDepth(const Mat &depth_truth, const Mat &depth_estimate, const Mat &depth_variance, const double max_variance) {
    //double ave_depth_error = 0;//average error
    double ave_depth_error_sq = 0;//squared error
    int cnt_depth_data = 0;

    #pragma omp parallel for reduction(+:ave_depth_error_sq) reduction(+:cnt_depth_data)
    for (int y = border; y < depth_truth.rows - border; y++)
        for (int x = border; x < depth_truth.cols - border; x++) 
        {
            const double variance = depth_variance.ptr<double>(y)[x];
            if(variance>=max_variance) continue; 
            const double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
            //ave_depth_error += error;
            ave_depth_error_sq += error * error;
            cnt_depth_data++;
        }
    //ave_depth_error /= cnt_depth_data;
    ave_depth_error_sq /= cnt_depth_data;

    //cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
    cout << "Average error (RMS) = " << sqrt(ave_depth_error_sq) << endl;  
}

void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

