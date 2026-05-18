//
// Created by xiang on 2022/1/21.
//

#include "ad/imu/gins_pre_integ.h"
#include "ad/imu/static_imu_init.h"
#include "ad/io/io_utils.h"
#include "utils/utm/utm_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

#include <filesystem>

#include <fstream>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iomanip>

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

/**
 * Run the GINS system based on pre-integration
 */
DEFINE_string(txt_path, dataDir + "/ad/gnss_nav/10.txt", "Data file path");
DEFINE_double(antenna_angle, 12.06,
              "RTK antenna installation yaw offset (degrees)");
DEFINE_double(antenna_pox_x, -0.17, "RTK antenna installation offset X");
DEFINE_double(antenna_pox_y, -0.20, "RTK antenna installation offset Y");
DEFINE_bool(with_ui, true, "Whether to display the graphical UI");
DEFINE_bool(debug, false, "Whether to print debug information");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (fLS::FLAGS_txt_path.empty()) {
    return -1;
  }

  // Initializer
  sad::StaticIMUInit imu_init; // Use default configuration

  sad::TxtIO io(fLS::FLAGS_txt_path);
  Vec2d antenna_pos(fLD::FLAGS_antenna_pox_x, fLD::FLAGS_antenna_pox_y);

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

  std::string resultsPath = resultsDir + "/ad/gnss_nav/gins_preintg.txt";
  std::string resultsPathDir =
      std::filesystem::path(resultsPath).parent_path().string();
  if (!std::filesystem::exists(resultsPathDir)) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::ofstream fout(resultsPath);

  bool imu_inited = false, gnss_inited = false;

  sad::GinsPreInteg::Options gins_options;
  gins_options.verbose_ = FLAGS_debug;
  sad::GinsPreInteg gins(gins_options);

  bool first_gnss_set = false;
  Vec3d origin = Vec3d::Zero();

  std::shared_ptr<sad::ui::AutonomousDrivingViz> ui = nullptr;
  if (FLAGS_with_ui) {
    ui = std::make_shared<sad::ui::AutonomousDrivingViz>();
    ui->Init();
  }

  /// Set various callback functions
  io.SetIMUProcessFunc([&](const sad::IMU &imu) {
      /// IMU processing function
      if (!imu_init.InitSuccess()) {
        imu_init.AddIMU(imu);
        return;
      }

      /// IMU initialization is required
      if (!imu_inited) {
        // Read initial biases and configure GINS
        sad::GinsPreInteg::Options options;
        options.preinteg_options_.init_bg_ = imu_init.GetInitBg();
        options.preinteg_options_.init_ba_ = imu_init.GetInitBa();
        options.gravity_ = imu_init.GetGravity();
        gins.SetOptions(options);
        imu_inited = true;
        return;
      }

      if (!gnss_inited) {
        /// Wait for valid RTK data
        return;
      }

      /// Start prediction only after GNSS is also received
      gins.AddImu(imu);

      auto state = gins.GetState();
      save_result(fout, state);
      if (ui) {
        ui->UpdateNavState(state);
        usleep(5e2);
      }
    })
      .SetGNSSProcessFunc([&](const sad::GNSS &gnss) {
        /// GNSS processing function
        if (!imu_inited) {
          return;
        }

        sad::GNSS gnss_convert = gnss;
        if (!sad::ConvertGps2UTM(gnss_convert, antenna_pos,
                                 FLAGS_antenna_angle) ||
            !gnss_convert.heading_valid_) {
          return;
        }

        /// Subtract the origin
        if (!first_gnss_set) {
          origin = gnss_convert.utm_pose_.translation();
          first_gnss_set = true;
        }
        gnss_convert.utm_pose_.translation() -= origin;

        gins.AddGnss(gnss_convert);

        auto state = gins.GetState();
        save_result(fout, state);
        if (ui) {
          ui->UpdateNavState(state);
          usleep(1e3);
        }
        gnss_inited = true;
      })
      .SetOdomProcessFunc([&](const sad::Odom &odom) {
        imu_init.AddOdom(odom);

        if (imu_inited && gnss_inited) {
          gins.AddOdom(odom);
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