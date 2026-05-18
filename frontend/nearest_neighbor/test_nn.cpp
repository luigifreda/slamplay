//
// Created by xiang on 2021/8/19.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>

#include "ad/common/sys_utils.h"
#include "ad/grid/gridnn.hpp"
#include "ad/grid/octo_tree.h"
#include "ad/kdtree/bfnn.h"
#include "ad/kdtree/kdtree.h"
#include "ad/pointcloud/point_cloud_utils.h"
#include "ad/pointcloud/point_types.h"

#include "macros.h"

std::string dataDir = STR(DATA_DIR); // DATA_DIR set by compilers flag

DEFINE_string(first_scan_path, dataDir + "/ad/pcd/first.pcd",
              "first point cloud path");
DEFINE_string(second_scan_path, dataDir + "/ad/pcd/second.pcd",
              "second point cloud path");
DEFINE_double(ANN_alpha, 1.0, "ANN scale factor");

TEST(CH5_TEST, BFNN) {
  sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
  pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
  pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

  if (first->empty() || second->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  // voxel grid to 0.05
  sad::VoxelGrid(first);
  sad::VoxelGrid(second);

  LOG(INFO) << "points: " << first->size() << ", " << second->size();

  // Evaluate single-threaded and multi-threaded brute-force matching
  sad::evaluate_and_call(
      [&first, &second]() {
        std::vector<std::pair<size_t, size_t>> matches;
        sad::bfnn_cloud(first, second, matches);
      },
      "Brute-force matching (single-threaded)", 5);
  sad::evaluate_and_call(
      [&first, &second]() {
        std::vector<std::pair<size_t, size_t>> matches;
        sad::bfnn_cloud_mt(first, second, matches);
      },
      "Brute-force matching (multi-threaded)", 5);

  SUCCEED();
}

/**
 * Evaluate nearest neighbor correctness
 * @param truth ground truth
 * @param esti  estimated matches
 */
void EvaluateMatches(const std::vector<std::pair<size_t, size_t>> &truth,
                     const std::vector<std::pair<size_t, size_t>> &esti) {
  int fp = 0; // false-positive: exists in esti but not in truth
  int fn = 0; // false-negative: exists in truth but not in esti

  LOG(INFO) << "truth: " << truth.size() << ", esti: " << esti.size();

  /// Check whether a match exists in the other container
  auto exist = [](const std::pair<size_t, size_t> &data,
                  const std::vector<std::pair<size_t, size_t>> &vec) -> bool {
    return std::find(vec.begin(), vec.end(), data) != vec.end();
  };

  int effective_esti = 0;
  for (const auto &d : esti) {
    if (d.first != sad::math::kINVALID_ID &&
        d.second != sad::math::kINVALID_ID) {
      effective_esti++;

      if (!exist(d, truth)) {
        fp++;
      }
    }
  }

  for (const auto &d : truth) {
    if (!exist(d, esti)) {
      fn++;
    }
  }

  float precision = 1.0 - float(fp) / effective_esti;
  float recall = 1.0 - float(fn) / truth.size();
  LOG(INFO) << "precision: " << precision << ", recall: " << recall
            << ", fp: " << fp << ", fn: " << fn;
}

TEST(CH5_TEST, GRID_NN) {
  sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
  pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
  pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

  if (first->empty() || second->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  // voxel grid to 0.05
  sad::VoxelGrid(first);
  sad::VoxelGrid(second);

  LOG(INFO) << "points: " << first->size() << ", " << second->size();

  std::vector<std::pair<size_t, size_t>> truth_matches;
  sad::bfnn_cloud(first, second, truth_matches);

  // Compare different types of grid
  sad::GridNN<2> grid0(0.1, sad::GridNN<2>::NearbyType::CENTER),
      grid4(0.1, sad::GridNN<2>::NearbyType::NEARBY4),
      grid8(0.1, sad::GridNN<2>::NearbyType::NEARBY8);
  sad::GridNN<3> grid3(0.1, sad::GridNN<3>::NearbyType::NEARBY6);

  grid0.SetPointCloud(first);
  grid4.SetPointCloud(first);
  grid8.SetPointCloud(first);
  grid3.SetPointCloud(first);

  // Evaluate various versions of Grid NN
  // sorry, no C++17 template lambda... the code below has to be a bit verbose
  LOG(INFO) << "===================";
  std::vector<std::pair<size_t, size_t>> matches;
  sad::evaluate_and_call(
      [&first, &second, &grid0, &matches]() {
        grid0.GetClosestPointForCloud(first, second, matches);
      },
      "Grid0 single-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid0, &matches]() {
        grid0.GetClosestPointForCloudMT(first, second, matches);
      },
      "Grid0 multi-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid4, &matches]() {
        grid4.GetClosestPointForCloud(first, second, matches);
      },
      "Grid4 single-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid4, &matches]() {
        grid4.GetClosestPointForCloudMT(first, second, matches);
      },
      "Grid4 multi-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid8, &matches]() {
        grid8.GetClosestPointForCloud(first, second, matches);
      },
      "Grid8 single-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid8, &matches]() {
        grid8.GetClosestPointForCloudMT(first, second, matches);
      },
      "Grid8 multi-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid3, &matches]() {
        grid3.GetClosestPointForCloud(first, second, matches);
      },
      "Grid 3D single-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  LOG(INFO) << "===================";
  sad::evaluate_and_call(
      [&first, &second, &grid3, &matches]() {
        grid3.GetClosestPointForCloudMT(first, second, matches);
      },
      "Grid 3D multi-threaded", 10);
  EvaluateMatches(truth_matches, matches);

  SUCCEED();
}

TEST(CH5_TEST, KDTREE_BASICS) {
  sad::CloudPtr cloud(new sad::PointCloudType);
  sad::PointType p1, p2, p3, p4;
  p1.x = 0;
  p1.y = 0;
  p1.z = 0;

  p2.x = 1;
  p2.y = 0;
  p2.z = 0;

  p3.x = 0;
  p3.y = 1;
  p3.z = 0;

  p4.x = 1;
  p4.y = 1;
  p4.z = 0;

  cloud->points.push_back(p1);
  cloud->points.push_back(p2);
  cloud->points.push_back(p3);
  cloud->points.push_back(p4);

  sad::KdTree kdtree;
  kdtree.BuildTree(cloud);
  kdtree.PrintAll();

  SUCCEED();
}

TEST(CH5_TEST, KDTREE_KNN) {
  sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
  pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
  pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

  if (first->empty() || second->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  // voxel grid to 0.05
  sad::VoxelGrid(first);
  sad::VoxelGrid(second);

  sad::KdTree kdtree;
  sad::evaluate_and_call([&first, &kdtree]() { kdtree.BuildTree(first); },
                         "Kd Tree build", 1);

  kdtree.SetEnableANN(true, FLAGS_ANN_alpha);

  LOG(INFO) << "Kd tree leaves: " << kdtree.size()
            << ", points: " << first->size();

  // Compare with bfnn
  std::vector<std::pair<size_t, size_t>> true_matches;
  sad::bfnn_cloud_mt_k(first, second, true_matches);

  // Run knn on the 2nd point cloud
  std::vector<std::pair<size_t, size_t>> matches;
  sad::evaluate_and_call(
      [&first, &second, &kdtree, &matches]() {
        kdtree.GetClosestPointMT(second, matches, 5);
      },
      "Kd Tree 5NN multi-threaded", 1);
  EvaluateMatches(true_matches, matches);

  LOG(INFO) << "building kdtree pcl";
  // Compare with PCL
  pcl::search::KdTree<sad::PointType> kdtree_pcl;
  sad::evaluate_and_call(
      [&first, &kdtree_pcl]() { kdtree_pcl.setInputCloud(first); },
      "Kd Tree build", 1);

  LOG(INFO) << "searching pcl";
  matches.clear();
  std::vector<int> search_indices(second->size());
  for (size_t i = 0; i < second->points.size(); i++) {
    search_indices[i] = i;
  }

  std::vector<std::vector<int>> result_index;
  std::vector<std::vector<float>> result_distance;
  sad::evaluate_and_call(
      [&]() {
        kdtree_pcl.nearestKSearch(*second, search_indices, 5, result_index,
                                  result_distance);
      },
      "Kd Tree 5NN in PCL", 1);
  for (size_t i = 0; i < second->points.size(); i++) {
    for (size_t j = 0; j < result_index[i].size(); ++j) {
      const int m = result_index[i][j];
      // double d = result_distance[i][j];
      matches.push_back({m, i});
    }
  }
  EvaluateMatches(true_matches, matches);

  LOG(INFO) << "done.";

  SUCCEED();
}

TEST(CH5_TEST, OCTREE_BASICS) {
  sad::CloudPtr cloud(new sad::PointCloudType);
  sad::PointType p1, p2, p3, p4;
  p1.x = 0;
  p1.y = 0;
  p1.z = 0;

  p2.x = 1;
  p2.y = 0;
  p2.z = 0;

  p3.x = 0;
  p3.y = 1;
  p3.z = 0;

  p4.x = 1;
  p4.y = 1;
  p4.z = 0;

  cloud->points.push_back(p1);
  cloud->points.push_back(p2);
  cloud->points.push_back(p3);
  cloud->points.push_back(p4);

  sad::OctoTree octree;
  octree.BuildTree(cloud);
  octree.SetApproximate(false);
  LOG(INFO) << "Octo tree leaves: " << octree.size()
            << ", points: " << cloud->size();

  SUCCEED();
}

TEST(CH5_TEST, OCTREE_KNN) {
  sad::CloudPtr first(new sad::PointCloudType), second(new sad::PointCloudType);
  pcl::io::loadPCDFile(FLAGS_first_scan_path, *first);
  pcl::io::loadPCDFile(FLAGS_second_scan_path, *second);

  if (first->empty() || second->empty()) {
    LOG(ERROR) << "cannot load cloud";
    FAIL();
  }

  // voxel grid to 0.05
  sad::VoxelGrid(first);
  sad::VoxelGrid(second);

  sad::OctoTree octree;
  sad::evaluate_and_call([&first, &octree]() { octree.BuildTree(first); },
                         "Octo Tree build", 1);

  octree.SetApproximate(true, FLAGS_ANN_alpha);
  LOG(INFO) << "Octo tree leaves: " << octree.size()
            << ", points: " << first->size();

  /// Test KNN
  LOG(INFO) << "testing knn";
  std::vector<std::pair<size_t, size_t>> matches;
  sad::evaluate_and_call(
      [&first, &second, &octree, &matches]() {
        octree.GetClosestPointMT(second, matches, 5);
      },
      "Octo Tree 5NN multi-threaded", 1);

  LOG(INFO) << "comparing with bfnn";
  /// Compare with ground truth
  std::vector<std::pair<size_t, size_t>> true_matches;
  sad::bfnn_cloud_mt_k(first, second, true_matches);
  EvaluateMatches(true_matches, matches);

  LOG(INFO) << "done.";

  SUCCEED();
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = google::INFO;
  FLAGS_colorlogtostderr = true;

  testing::InitGoogleTest(&argc, argv);
  google::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
