#include <gflags/gflags.h>
#include <glog/logging.h>

#include "lio_slam/frontend.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(config_yaml, configDir + "/lio_slam/mapping.yaml", "Config file");

// Test frontend pipeline
int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "testing frontend";
  sad::Frontend frontend(FLAGS_config_yaml);
  if (!frontend.Init()) {
    LOG(ERROR) << "failed to init frontend.";
    return -1;
  }

  frontend.Run();
  return 0;
}