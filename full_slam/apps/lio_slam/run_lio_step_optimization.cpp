//
// Created by xiang on 22-12-7.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "lio_slam/optimization.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(config_yaml, configDir + "/lio_slam/mapping.yaml", "Config file");
DEFINE_int64(stage, 1, "Run stage 1 or stage 2 optimization");

// Test optimization pipeline
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  assert(FLAGS_stage == 1 || FLAGS_stage == 2);

  LOG(INFO) << "testing optimization";
  sad::Optimization opti(FLAGS_config_yaml);
  if (!opti.Init(FLAGS_stage)) {
    LOG(ERROR) << "failed to init frontend.";
    return -1;
  }

  opti.Run();
  return 0;
}