//
// Created by xiang on 2021/9/27.
//

#include "ad/grid/octo_tree.h"
#include "ad/common/math_utils.h"

#include <execution>

namespace sad {

bool OctoTree::BuildTree(const CloudPtr &cloud) {
  if (cloud->empty()) {
    return false;
  }

  cloud_.clear();
  cloud_.resize(cloud->size());
  for (size_t i = 0; i < cloud->points.size(); ++i) {
    cloud_[i] = ToVec3f(cloud->points[i]);
  }

  Clear();
  Reset();

  IndexVec idx(cloud->size());
  for (int i = 0; i < cloud->points.size(); ++i) {
    idx[i] = i;
  }

  // Generate bounding box for the root node
  root_->box_ = ComputeBoundingBox();
  Insert(idx, root_.get());

  return true;
}

void OctoTree::Insert(const IndexVec &points, OctoTreeNode *node) {
  nodes_.insert({node->id_, node});

  if (points.empty()) {
    return;
  }

  if (points.size() == 1) {
    size_++;
    node->point_idx_ = points[0];
    return;
  }

  /// As long as the point count is not 1, keep expanding this node
  std::vector<IndexVec> children_points;
  ExpandNode(node, points, children_points);

  /// Insert into child nodes
  for (size_t i = 0; i < 8; ++i) {
    Insert(children_points[i], node->children[i]);
  }
}

void OctoTree::ExpandNode(OctoTreeNode *node, const IndexVec &parent_idx,
                          std::vector<IndexVec> &children_idx) {
  children_idx.resize(8);
  for (int i = 0; i < 8; ++i) {
    node->children[i] = new OctoTreeNode();
    node->children[i]->id_ = tree_node_id_++;
  }

  const Box3D &b = node->box_; // this node's box
  // center point
  float c_x = 0.5 * (node->box_.min_[0] + node->box_.max_[0]);
  float c_y = 0.5 * (node->box_.min_[1] + node->box_.max_[1]);
  float c_z = 0.5 * (node->box_.min_[2] + node->box_.max_[2]);

  // Diagram of 8 sub-boxes
  // clang-format off
    // Layer 1: top-left 1, top-right 2, bottom-left 3, bottom-right 4
    // Layer 2: top-left 5, top-right 6, bottom-left 7, bottom-right 8
    //     ---> x    /-------/-------/|
    //    /|        /-------/-------/||
    //   / |       /-------/-------/ ||
    //  y  |z      |       |       | /|
    //             |_______|_______|/|/
    //             |       |       | /
    //             |_______|_______|/
  // clang-format on
  node->children[0]->box_ = {b.min_[0], c_x, b.min_[1], c_y, b.min_[2], c_z};
  node->children[1]->box_ = {c_x, b.max_[0], b.min_[1], c_y, b.min_[2], c_z};
  node->children[2]->box_ = {b.min_[0], c_x, c_y, b.max_[1], b.min_[2], c_z};
  node->children[3]->box_ = {c_x, b.max_[0], c_y, b.max_[1], b.min_[2], c_z};

  node->children[4]->box_ = {b.min_[0], c_x, b.min_[1], c_y, c_z, b.max_[2]};
  node->children[5]->box_ = {c_x, b.max_[0], b.min_[1], c_y, c_z, b.max_[2]};
  node->children[6]->box_ = {b.min_[0], c_x, c_y, b.max_[1], c_z, b.max_[2]};
  node->children[7]->box_ = {c_x, b.max_[0], c_y, b.max_[1], c_z, b.max_[2]};

  // Assign points to child nodes
  for (int idx : parent_idx) {
    const auto pt = cloud_[idx];
    for (int i = 0; i < 8; ++i) {
      if (node->children[i]->box_.Inside(pt)) {
        children_idx[i].emplace_back(idx);
        break;
      }
    }
  }
}

Box3D OctoTree::ComputeBoundingBox() {
  float min_values[3] = {std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max()};
  float max_values[3] = {-std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max(),
                         -std::numeric_limits<float>::max()};

  for (const auto &p : cloud_) {
    for (int i = 0; i < 3; ++i) {
      max_values[i] = p[i] > max_values[i] ? p[i] : max_values[i];
      min_values[i] = p[i] < min_values[i] ? p[i] : min_values[i];
    }
  }

  return {min_values[0], max_values[0], min_values[1],
          max_values[1], min_values[2], max_values[2]};
}

bool OctoTree::GetClosestPoint(const PointType &pt,
                               std::vector<int> &closest_idx, int k) const {
  if (k > size_) {
    LOG(ERROR) << "cannot set k larger than cloud size: " << k << ", " << size_;
    return false;
  }

  std::priority_queue<NodeAndDistanceOcto> knn_result;
  Knn(ToVec3f(pt), root_.get(), knn_result);

  // Sort and return results
  closest_idx.resize(knn_result.size());
  for (int i = closest_idx.size() - 1; i >= 0; --i) {
    // Insert in reverse order
    closest_idx[i] = knn_result.top().node_->point_idx_;
    knn_result.pop();
  }
  return true;
}

bool OctoTree::GetClosestPointMT(
    const CloudPtr &cloud, std::vector<std::pair<size_t, size_t>> &matches,
    int k) {
  k_ = k;
  matches.resize(cloud->size() * k);
  // Index
  std::vector<int> index(cloud->size());
  for (int i = 0; i < cloud->points.size(); ++i) {
    index[i] = i;
  }

  std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                [this, &cloud, &matches, &k](int idx) {
                  std::vector<int> closest_idx;
                  GetClosestPoint(cloud->points[idx], closest_idx, k);

                  for (int i = 0; i < k; ++i) {
                    matches[idx * k + i].second = idx;
                    if (i < closest_idx.size()) {
                      matches[idx * k + i].first = closest_idx[i];
                    } else {
                      matches[idx * k + i].first = math::kINVALID_ID;
                    }
                  }
                });

  return true;
}

void OctoTree::Clear() {
  for (const auto &np : nodes_) {
    if (np.second != root_.get()) {
      delete np.second;
    }
  }

  root_ = nullptr;
  size_ = 0;
  tree_node_id_ = 0;
}

void OctoTree::Reset() {
  tree_node_id_ = 0;
  root_.reset(new OctoTreeNode());
  root_->id_ = tree_node_id_++;
  size_ = 0;
}

void OctoTree::Knn(const Vec3f &pt, OctoTreeNode *node,
                   std::priority_queue<NodeAndDistanceOcto> &knn_result) const {
  if (node->IsLeaf()) {
    if (node->point_idx_ != -1) {
      // If it's a leaf, check whether this point is a nearest neighbor
      ComputeDisForLeaf(pt, node, knn_result);
      return;
    }
    return;
  }

  // Check which cell pt falls in; prioritize searching the subtree containing
  // pt, then check if other subtrees need to be searched. If pt is outside,
  // prioritize searching the nearest subtree.
  int idx_child = -1;
  float min_dis = std::numeric_limits<float>::max();
  for (int i = 0; i < 8; ++i) {
    if (node->children[i]->box_.Inside(pt)) {
      idx_child = i;
      break;
    } else {
      float d = node->children[i]->box_.Dis(pt);
      if (d < min_dis) {
        idx_child = i;
        min_dis = d;
      }
    }
  }

  // Check idx_child first
  Knn(pt, node->children[idx_child], knn_result);

  // Then check the others
  for (int i = 0; i < 8; ++i) {
    if (i == idx_child) {
      continue;
    }

    if (NeedExpand(pt, node->children[i], knn_result)) {
      Knn(pt, node->children[i], knn_result);
    }
  }
}

void OctoTree::ComputeDisForLeaf(
    const Vec3f &pt, OctoTreeNode *node,
    std::priority_queue<NodeAndDistanceOcto> &knn_result) const {
  // Compare with the result queue; insert if closer than the farthest distance
  float dis2 = Dis2(pt, cloud_[node->point_idx_]);
  if (knn_result.size() < k_) {
    // results has fewer than k entries
    knn_result.push({node, dis2});
  } else {
    // results has k entries, compare current distance with the farthest
    if (dis2 < knn_result.top().distance_) {
      knn_result.push({node, dis2});
      knn_result.pop();
    }
  }
}

bool OctoTree::NeedExpand(
    const Vec3f &pt, OctoTreeNode *node,
    std::priority_queue<NodeAndDistanceOcto> &knn_result) const {
  if (knn_result.size() < k_) {
    return true;
  }

  if (approximate_) {
    float d = node->box_.Dis(pt);
    if ((d * d) < knn_result.top().distance_ * alpha_) {
      return true;
    } else {
      return false;
    }
  } else {
    // When not using ANN, search normally
    float d = node->box_.Dis(pt);
    if ((d * d) < knn_result.top().distance_) {
      return true;
    } else {
      return false;
    }
  }
}

} // namespace sad
