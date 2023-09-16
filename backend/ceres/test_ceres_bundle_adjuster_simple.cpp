// A minimal, self-contained bundle adjuster using Ceres, that reads
// files from University of Washington' Bundle Adjustment in the Large dataset:
// http://grail.cs.washington.edu/projects/bal
//
// This does not use the best configuration for solving; see the more involved
// bundle_adjuster.cc file for details.
// https://ceres-solver.googlesource.com/ceres-solver/+/master/examples/bundle_adjuster.cc 


#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); //DATA_DIR set by compilers flag 
std::string bal_file = dataDir + "/bal/ladybug49.txt";

// Read a Bundle Adjustment in the Large dataset.
class BALProblem {
 public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] parameters_;
  }
  int num_observations() const { return num_observations_; }
  const double* observations() const { return observations_; }
  double* mutable_cameras() { return parameters_; }
  double* mutable_points() { return parameters_ + 9 * num_cameras_; }
  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 9;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }
  bool LoadFile(const char* filename) {
    FILE* fptr = fopen(filename, "r");
    if (fptr == nullptr) {
      return false;
    };
    FscanfOrDie(fptr, "%d", &num_cameras_);
    FscanfOrDie(fptr, "%d", &num_points_);
    FscanfOrDie(fptr, "%d", &num_observations_);
    point_index_ = new int[num_observations_];
    camera_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];
    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];
    for (int i = 0; i < num_observations_; ++i) {
      FscanfOrDie(fptr, "%d", camera_index_ + i);
      FscanfOrDie(fptr, "%d", point_index_ + i);
      for (int j = 0; j < 2; ++j) {
        FscanfOrDie(fptr, "%lf", observations_ + 2 * i + j);
      }
    }
    for (int i = 0; i < num_parameters_; ++i) {
      FscanfOrDie(fptr, "%lf", parameters_ + i);
    }
    return true;
  }

 private:
  template <typename T>
  void FscanfOrDie(FILE* fptr, const char* format, T* value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
      LOG(FATAL) << "Invalid UW data file.";
    }
  }

  int num_cameras_;
  int num_points_;
  int num_observations_;
  int num_parameters_;
  int* point_index_;
  int* camera_index_;
  double* observations_;
  double* parameters_;
};


// Templated pinhole camera model for used with Ceres.  The camera is
// parameterized using 9 parameters: 3 for rotation, 3 for translation, 1 for
// focal length and 2 for radial distortion. The principal point is not modeled
// (i.e. it is assumed be located at the image center).
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}
  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p); // rotate point 
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; // add translation after rotation 
    p[1] += camera[4];
    p[2] += camera[5];
    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = -p[0] / p[2];
    T yp = -p[1] / p[2];
    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y) {
    return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
        new SnavelyReprojectionError(observed_x, observed_y)));
  }
  double observed_x;
  double observed_y;
};


int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);

  std::string input_file; 
#if 0
  if (argc != 2) {
    std::cerr << "usage: simple_bundle_adjuster <bal_problem>\n";
    return 1;
  }
  input_file = argv[1];
#else
  input_file = bal_file;
#endif 

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(input_file.c_str())) {
    std::cerr << "ERROR: unable to open file " << input_file << "\n";
    return 1;
  }
  const double* observations = bal_problem.observations();
  // Create residuals for each observation in the bundle adjustment problem. The
  // parameters for cameras and points are added automatically.

  ceres::Problem problem;
  
  for (int i = 0; i < bal_problem.num_observations(); ++i) {
    // Each Residual block takes a point and a camera as input and outputs a 2
    // dimensional residual. Internally, the cost function stores the observed
    // image location and compares the reprojection against the observation.
    ceres::CostFunction* cost_function = SnavelyReprojectionError::Create(
        observations[2 * i + 0], observations[2 * i + 1]);

#if 1
    ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0); 
    //ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);    
#else
    ceres::LossFunction* loss_function = nullptr; /* squared loss */
#endif 

    problem.AddResidualBlock(cost_function,
                             loss_function,
                             bal_problem.mutable_camera_for_observation(i),
                             bal_problem.mutable_point_for_observation(i));
  }

  // Make Ceres automatically detect the bundle structure. Note that the
  // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
  // for standard bundle adjustment problems.
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  return 0;
}