#include "lio_slam/loop_closure.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "macros.h"

std::string dataDir = STR(DATA_DIR);       // DATA_DIR set by compilers flag
std::string resultsDir = STR(RESULTS_DIR); // RESULTS_DIR set by compilers flag
std::string configDir = STR(CONFIG_DIR);   // CONFIG_DIR set by compilers flag

DEFINE_string(config_yaml, configDir + "/lio_slam/mapping.yaml", "Config file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags(&argc, &argv, true);

  sad::LoopClosure lc(FLAGS_config_yaml);
  lc.Init();
  lc.Run();

  return 0;
}