#include <filesystem>
#include <glog/logging.h>
#include <iomanip>
#include <memory>

#include "ad/io/io_utils.h"
#include "ad/pointcloud/point_types.h"
#include "utils/utm/utm_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(txt_path, dataDir + "/ad/gnss_nav/10.txt",
              "Path to the data file");

// The following parameters are specific to the dataset provided with this book.
DEFINE_double(antenna_angle, 12.06,
              "RTK antenna mounting yaw offset (degrees)");
DEFINE_double(antenna_pox_x, -0.17, "RTK antenna mounting offset X");
DEFINE_double(antenna_pox_y, -0.20, "RTK antenna mounting offset Y");
DEFINE_bool(with_ui, true, "Whether to display the graphical interface");

/**
 * This program demonstrates how to process GNSS data.
 * It converts raw GNSS readings into a 6-DoF pose suitable for downstream
 * processing. The pipeline includes UTM conversion, RTK antenna extrinsics, and
 * coordinate-frame conversion.
 *
 * The results are written to a file and then visualized with a Python script.
 */

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (fLS::FLAGS_txt_path.empty()) {
    return -1;
  }

  sad::TxtIO io(fLS::FLAGS_txt_path);

  std::string resultsPath = resultsDir + "/ad/gnss_nav/gnss_output.txt";
  std::string resultsPathDir =
      std::filesystem::path(resultsPath).parent_path().string();
  if (!std::filesystem::exists(resultsPathDir)) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::ofstream fout(resultsPath);

  Vec2d antenna_pos(FLAGS_antenna_pox_x, FLAGS_antenna_pox_y);

  // lambda function to save the result
  auto save_result = [](std::ofstream &fout, double timestamp,
                        const SE3 &pose) {
    auto save_vec3 = [](std::ofstream &fout, const Vec3d &v) {
      fout << v[0] << " " << v[1] << " " << v[2] << " ";
    };
    auto save_quat = [](std::ofstream &fout, const Quatd &q) {
      fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    fout << std::setprecision(18) << timestamp << " " << std::setprecision(9);
    save_vec3(fout, pose.translation());
    save_quat(fout, pose.unit_quaternion());
    fout << std::endl;
  };

  std::shared_ptr<sad::ui::AutonomousDrivingViz> ui = nullptr;
  if (FLAGS_with_ui) {
    ui = std::make_shared<sad::ui::AutonomousDrivingViz>();
    ui->Init();
  }

  bool first_gnss_set = false;
  Vec3d origin = Vec3d::Zero();
  io.SetGNSSProcessFunc([&](const sad::GNSS &gnss) {
      sad::GNSS gnss_out = gnss;
      if (sad::ConvertGps2UTM(gnss_out, antenna_pos, FLAGS_antenna_angle)) {
        if (!first_gnss_set) {
          origin = gnss_out.utm_pose_.translation();
          first_gnss_set = true;
        }

        /// Subtract the chosen origin.
        gnss_out.utm_pose_.translation() -= origin;

        save_result(fout, gnss_out.unix_time_, gnss_out.utm_pose_);

        if (ui) {
          ui->UpdateNavState(sad::NavStated(gnss_out.unix_time_,
                                            gnss_out.utm_pose_.so3(),
                                            gnss_out.utm_pose_.translation()));
          usleep(1e3);
        }
      } else {
        LOG(ERROR) << "Failed to convert GPS to UTM";
      }
    }).Go();

  if (ui) {
    while (!ui->ShouldQuit()) {
      usleep(1e5);
    }
    ui->Quit();
  }

  return 0;
}