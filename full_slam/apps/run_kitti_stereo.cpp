//
// Created by gaoxiang on 19-5-4.
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay
//

#include <gflags/gflags.h>
#include "io/messages.h"
#include "macros.h"
#include "myslam/visual_odometry.h"

DEFINE_string(config_file, "../config/kitti.yaml", "config file path");

int main(int argc, char **argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry(FLAGS_config_file, myslam::DatasetType::KITTI));
    MSG_ASSERT(vo->Init() == true, "VO didn't init correctly");
    vo->Run();

    return 0;
}
