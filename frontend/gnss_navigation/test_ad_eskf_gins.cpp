#include "ad/imu/static_imu_init.h"
#include "ad/io/io_utils.h"
#include "ad/kf/eskf.hpp"
#include "utils/utm/utm_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

#include <filesystem>
#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>
#include <memory>

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(txt_path, dataDir + "/ad/gnss_nav/10.txt",
              "Path to the data file");
DEFINE_double(antenna_angle, 12.06,
              "RTK antenna mounting yaw offset (degrees)");
DEFINE_double(antenna_pox_x, -0.17, "RTK antenna mounting offset X");
DEFINE_double(antenna_pox_y, -0.20, "RTK antenna mounting offset Y");
DEFINE_bool(with_ui, true, "Whether to display the graphical interface");
DEFINE_bool(with_odom, false, "Whether to include wheel odometry");

/**
 * This program demonstrates integrated navigation using RTK and IMU.
 */
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (fLS::FLAGS_txt_path.empty()) {
    return -1;
  }

  // Initializer.
  sad::StaticIMUInit imu_init; // Use the default configuration.

  // ESKF
  sad::ESKFD eskf;

  sad::TxtIO io(FLAGS_txt_path);
  Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

  // lambda functions to save the result
  auto save_vec3 = [](std::ofstream &fout, const Vec3d &v) {
    fout << v[0] << " " << v[1] << " " << v[2] << " ";
  };
  auto save_quat = [](std::ofstream &fout, const Quatd &q) {
    fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
  };

  auto save_result = [&save_vec3,
                      &save_quat](std::ofstream &fout,
                                  const sad::NavStated &save_state) {
    fout << std::setprecision(18) << save_state.timestamp_ << " "
         << std::setprecision(9);
    save_vec3(fout, save_state.p_);
    save_quat(fout, save_state.R_.unit_quaternion());
    save_vec3(fout, save_state.v_);
    save_vec3(fout, save_state.bg_);
    save_vec3(fout, save_state.ba_);
    fout << std::endl;
  };

  std::string resultsPath = resultsDir + "/ad/gnss_nav/gins.txt";
  std::string resultsPathDir =
      std::filesystem::path(resultsPath).parent_path().string();
  if (!std::filesystem::exists(resultsPathDir)) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::ofstream fout(resultsPath);

  bool imu_inited = false, gnss_inited = false;

  std::shared_ptr<sad::ui::AutonomousDrivingViz> ui = nullptr;
  if (FLAGS_with_ui) {
    ui = std::make_shared<sad::ui::AutonomousDrivingViz>();
    ui->Init();
  }

  /// Set the processing callbacks.
  bool first_gnss_set = false;
  Vec3d origin = Vec3d::Zero();

  io.SetIMUProcessFunc([&](const sad::IMU &imu) {
      /// IMU processing callback.
      if (!imu_init.InitSuccess()) {
        imu_init.AddIMU(imu);
        return;
      }

      /// Initialize the IMU first.
      if (!imu_inited) {
        // Read the initial biases and configure the ESKF.
        sad::ESKFD::Options options;
        // The noise values are estimated by the initializer.
        options.gyro_var_ = sqrt(imu_init.GetCovGyro()[0]);
        options.acce_var_ = sqrt(imu_init.GetCovAcce()[0]);
        eskf.SetInitialConditions(options, imu_init.GetInitBg(),
                                  imu_init.GetInitBa(), imu_init.GetGravity());
        imu_inited = true;
        return;
      }

      if (!gnss_inited) {
        /// Wait for valid RTK data.
        return;
      }

      /// Start prediction only after GNSS has also been initialized.
      eskf.Predict(imu);

      /// `Predict` updates the ESKF, so the state can be published now.
      auto state = eskf.GetNominalState();
      if (ui) {
        ui->UpdateNavState(state);
      }

      /// Save data for plotting.
      save_result(fout, state);

      usleep(1e3);
    })
      .SetGNSSProcessFunc([&](const sad::GNSS &gnss) {
        /// GNSS processing callback.
        if (!imu_inited) {
          return;
        }

        sad::GNSS gnss_convert = gnss;
        if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos,
                                 FLAGS_antenna_angle) ||
            !gnss_convert.heading_valid_) {
          return;
        }

        /// Subtract the origin.
        if (!first_gnss_set) {
          origin = gnss_convert.utm_pose_.translation();
          first_gnss_set = true;
        }
        gnss_convert.utm_pose_.translation() -= origin;

        // RTK heading must be valid before it can be fused into the ESKF.
        eskf.ObserveGps(gnss_convert);

        auto state = eskf.GetNominalState();
        if (ui) {
          ui->UpdateNavState(state);
        }
        save_result(fout, state);

        gnss_inited = true;
      })
      .SetOdomProcessFunc([&](const sad::Odom &odom) {
        /// Odometry processing callback. In this chapter, odometry is used only
        /// for initialization.
        imu_init.AddOdom(odom);
        if (FLAGS_with_odom && imu_inited && gnss_inited) {
          eskf.ObserveWheelSpeed(odom);
        }
      })
      .Go();

  while (ui && !ui->ShouldQuit()) {
    usleep(1e5);
  }
  if (ui) {
    ui->Quit();
  }
  return 0;
}