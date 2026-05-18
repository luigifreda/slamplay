#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ad/common/eigen_types.h"
#include "ad/common/math_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

/// This example demonstrates a vehicle moving in a circle.
/// The vehicle's angular and linear velocities can be configured via flags.

DEFINE_double(angular_velocity, 10.0, "Angular velocity in degrees");
DEFINE_double(linear_velocity, 5.0, "Vehicle forward linear velocity in m/s");
DEFINE_bool(use_quaternion, false,
            "Whether to use quaternion-based computation");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  /// Visualization
  sad::ui::AutonomousDrivingViz ui;
  if (ui.Init() == false) {
    return -1;
  }

  double angular_velocity_rad =
      FLAGS_angular_velocity *
      sad::math::kDEG2RAD;                   // Angular velocity in radians
  SE3 pose;                                  // Pose represented as T_wb
  Vec3d omega(0, 0, angular_velocity_rad);   // Angular velocity vector
  Vec3d v_body(FLAGS_linear_velocity, 0, 0); // Velocity in the body frame
  const double dt = 0.05;                    // Time step for each update

  while (ui.ShouldQuit() == false) {

    // Update position by using simple integration
    Vec3d v_world = pose.so3() * v_body;
    pose.translation() += v_world * dt;

    // Update rotation
    if (FLAGS_use_quaternion) {
      Quatd q = pose.unit_quaternion() * Quatd(1, 0.5 * omega[0] * dt,
                                               0.5 * omega[1] * dt,
                                               0.5 * omega[2] * dt);
      q.normalize();
      pose.so3() = SO3(q);
    } else {
      pose.so3() = pose.so3() * SO3::exp(omega * dt);
    }

    LOG(INFO) << "pose: " << pose.translation().transpose();
    ui.UpdateNavState(sad::NavStated(0, pose, v_world));

    usleep(dt * 1e6);
  }

  ui.Quit();
  return 0;
}