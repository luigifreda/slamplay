#include <filesystem>
#include <glog/logging.h>
#include <iomanip>

#include "ad/imu/imu_integration.h"
#include "ad/io/io_utils.h"
#include "viz/ad/autonomous_driving_viz.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag

DEFINE_string(imu_txt_path, dataDir + "/ad/gnss_nav/10.txt",
              "Path to the data file");
DEFINE_bool(with_ui, true, "Whether to display the graphical interface");

/**
 * This program demonstrates how to directly integrate IMU measurements.
 * It takes a text file under `data/ad/gnss_nav/` as input, writes the states to
 * `results/ad/gnss_nav/10.txt`, and also visualizes the vehicle motion in the
 * UI.
 */
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_imu_txt_path.empty()) {
    return -1;
  }

  sad::TxtIO io(FLAGS_imu_txt_path);

  // In this experiment, we assume the biases are known.
  Vec3d gravity(0, 0, -9.8); // Gravity direction.
  Vec3d init_bg(00.000224886, -7.61038e-05, -0.000742259);
  Vec3d init_ba(-0.165205, 0.0926887, 0.0058049);

  sad::IMUIntegration imu_integ(gravity, init_bg, init_ba);

  std::shared_ptr<sad::ui::AutonomousDrivingViz> ui = nullptr;
  if (FLAGS_with_ui) {
    ui = std::make_shared<sad::ui::AutonomousDrivingViz>();
    ui->Init();
  }

  /// Save the results.
  auto save_result = [](std::ofstream &fout, double timestamp,
                        const Sophus::SO3d &R, const Vec3d &v, const Vec3d &p) {
    auto save_vec3 = [](std::ofstream &fout, const Vec3d &v) {
      fout << v[0] << " " << v[1] << " " << v[2] << " ";
    };
    auto save_quat = [](std::ofstream &fout, const Quatd &q) {
      fout << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " ";
    };

    fout << std::setprecision(18) << timestamp << " " << std::setprecision(9);
    save_vec3(fout, p);
    save_quat(fout, R.unit_quaternion());
    save_vec3(fout, v);
    fout << std::endl;
  };

  std::string resultsPath =
      resultsDir + "/ad/gnss_nav/imu_direct_integration.txt";
  std::string resultsPathDir =
      std::filesystem::path(resultsPath).parent_path().string();
  if (!std::filesystem::exists(resultsPathDir)) {
    std::filesystem::create_directories(resultsPathDir);
  }
  std::ofstream fout(resultsPath);

  io.SetIMUProcessFunc([&imu_integ, &save_result, &fout,
                        &ui](const sad::IMU &imu) {
      imu_integ.AddIMU(imu);
      save_result(fout, imu.timestamp_, imu_integ.GetR(), imu_integ.GetV(),
                  imu_integ.GetP());
      if (ui) {
        ui->UpdateNavState(imu_integ.GetNavState());
        usleep(1e2);
      }
    }).Go();

  // If visualization is enabled, wait until the window is closed.
  while (ui && !ui->ShouldQuit()) {
    usleep(1e4);
  }

  if (ui) {
    ui->Quit();
  }

  return 0;
}