#include <experimental/filesystem>
#include <iostream>

#include <SESync/SESync.h>
#include <SESync/SESyncVisualizer.h>
#include <SESync/SESync_utils.h>

using namespace std;
using namespace SESync;

bool write_poses = false;

int main(int argc, char **argv) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
    exit(1);
  }

  size_t num_poses;
  measurements_t measurements = read_g2o_file(argv[1], num_poses);
  cout << "Loaded " << measurements.size() << " measurements between "
       << num_poses << " poses from file " << argv[1] << endl
       << endl;
  if (measurements.size() == 0) {
    cout << "Error: No measurements were read!"
         << " Are you sure the file exists?" << endl;
    exit(1);
  }

  SESyncOpts opts;
  opts.verbose = true;  // Print output to stdout
  opts.num_threads = 4;
  opts.initialization = Initialization::Random;  // Make it interesting.

  VisualizationOpts vopts;
  vopts.delay = 0.1;  // [s]
  vopts.img_name = "custom-img-name";
  vopts.img_dir =
      "sesync-iters-" +
      std::string(std::experimental::filesystem::path(argv[1]).filename());

  // Run SE-Sync, and launch the visualization magic.
  SESyncVisualizer viz(num_poses, measurements, opts, vopts);
  viz.RenderSynchronization();
}
