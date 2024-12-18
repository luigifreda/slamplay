// *************************************************************************
/* 
 * This file is part of the slamplay project.
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version, at your option. If this file is a modified/adapted 
 * version of an original file distributed under a different license that 
 * is not compatible with the GNU General Public License, the 
 * BSD 3-Clause License will apply instead.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */
// *************************************************************************
#include <SESync/SESync.h>
#include <SESync/SESync_utils.h>

#include <fstream>

#ifdef GPERFTOOLS
#include <gperftools/profiler.h>
#endif

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
  opts.verbose = true; // Print output to stdout

  // Initialization method
  // Options are:  Chordal, Random
  opts.initialization = Initialization::Chordal;

  // Specific form of the synchronization problem to solve
  // Options are: Simplified, Explicit, SOSync
  opts.formulation = Formulation::Simplified;

  // Initial
  opts.num_threads = 4;

#ifdef GPERFTOOLS
  ProfilerStart("SE-Sync.prof");
#endif

  /// RUN SE-SYNC!
  SESyncResult results = SESync::SESync(measurements, opts);

#ifdef GPERFTOOLS
  ProfilerStop();
#endif

  if (write_poses) {
    // Write output
    string filename = "poses.txt";
    cout << "Saving final poses to file: " << filename << endl;
    ofstream poses_file(filename);
    poses_file << results.xhat;
    poses_file.close();
  }
}
