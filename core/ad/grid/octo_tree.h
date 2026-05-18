//
// Created by xiang on 2021/9/27.
//

#ifndef SLAM_IN_AUTO_DRIVING_OCTO_TREE_H
#define SLAM_IN_AUTO_DRIVING_OCTO_TREE_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include <glog/logging.h>
#include <map>
#include <queue>

namespace sad {

// 3D Box storing min/max values along each axis
struct Box3D {
  Box3D() = default;
  Box3D(float min_x, float max_x, float min_y, float max_y, float min_z,
        float max_z)
      : min_{min_x, min_y, min_z}, max_{max_x, max_y, max_z} {}

  /// Check whether pt is inside
  bool Inside(const Vec3f &pt) const {
    return pt[0] <= max_[0] && pt[0] >= min_[0] && pt[1] <= max_[1] &&
           pt[1] >= min_[1] && pt[2] <= max_[2] && pt[2] >= min_[2];
  }

  /// Distance from point to 3D Box
  /// We take the max distance from the exterior point to the boundary
  float Dis(const Vec3f &pt) const {
    float ret = 0;
    for (int i = 0; i < 3; ++i) {
      if (pt[i] < min_[i]) {
        float d = min_[i] - pt[i];
        ret = d > ret ? d : ret;
      } else if (pt[i] > max_[i]) {
        float d = pt[i] - max_[i];
        ret = d > ret ? d : ret;
      }
    }

    assert(ret >= 0);
    return ret;
  }

  float min_[3] = {0};
  float max_[3] = {0};
};

/// Octo tree node
struct OctoTreeNode {
  int id_ = -1;
  int point_idx_ = -1;                   // point index, -1 means invalid
  Box3D box_;                            // bounding box
  OctoTreeNode *children[8] = {nullptr}; // child nodes

  bool IsLeaf() const {
    for (const OctoTreeNode *n : children) {
      if (n != nullptr) {
        return false;
      }
    }
    return true;
  }
};

/// Used to store knn results
struct NodeAndDistanceOcto {
  NodeAndDistanceOcto(OctoTreeNode *node, float dis2)
      : node_(node), distance_(dis2) {}
  OctoTreeNode *node_ = nullptr;
  float distance_ = 0; // squared distance, used for comparison

  bool operator<(const NodeAndDistanceOcto &other) const {
    return distance_ < other.distance_;
  }
};

class OctoTree {
public:
  explicit OctoTree() = default;
  ~OctoTree() { Clear(); }

  bool BuildTree(const CloudPtr &cloud);

  /// Get k nearest neighbors
  bool GetClosestPoint(const PointType &pt, std::vector<int> &closest_idx,
                       int k = 5) const;

  /// Find nearest neighbors for point cloud in parallel
  bool GetClosestPointMT(const CloudPtr &cloud,
                         std::vector<std::pair<size_t, size_t>> &match,
                         int k = 5);

  /// Set approximate nearest neighbor parameters
  void SetApproximate(bool use_ann = true, float alpha = 0.1) {
    approximate_ = use_ann;
    alpha_ = alpha;
  }

  /// Return the number of nodes
  size_t size() const { return size_; }

  /// Clear data
  void Clear();

private:
  /// Tree construction
  /**
   * Insert points at node
   * @param points
   * @param node
   */
  void Insert(const IndexVec &points, OctoTreeNode *node);

  /// Generate bounding box for the entire point cloud
  Box3D ComputeBoundingBox();

  /**
   * Expand a node into children
   * @param [in] node the node to expand
   * @param [in] parent_idx point cloud indices of the parent node
   * @param [out] children_idx point cloud indices of the child nodes
   */
  void ExpandNode(OctoTreeNode *node, const IndexVec &parent_idx,
                  std::vector<IndexVec> &children_idx);

  void Reset();

  /// Squared distance between two points
  static inline float Dis2(const Vec3f &p1, const Vec3f &p2) {
    return (p1 - p2).squaredNorm();
  }

  // Knn related
  /**
   * Check knn for the given point on an octo tree node, called recursively
   * @param pt     query point
   * @param node   octo tree node
   */
  void Knn(const Vec3f &pt, OctoTreeNode *node,
           std::priority_queue<NodeAndDistanceOcto> &result) const;

  /**
   * For a leaf node, compute its distance to the query point and try to insert
   * into results
   * @param pt    query point
   * @param node  octo tree node
   */
  void
  ComputeDisForLeaf(const Vec3f &pt, OctoTreeNode *node,
                    std::priority_queue<NodeAndDistanceOcto> &result) const;

  /**
   * Check whether the subtree under node needs to be expanded
   * @param pt   query point
   * @param node octo tree node
   * @return true if expansion is needed
   */
  bool NeedExpand(const Vec3f &pt, OctoTreeNode *node,
                  std::priority_queue<NodeAndDistanceOcto> &knn_result) const;

  int k_ = 5;                                    // knn neighbor count
  std::shared_ptr<OctoTreeNode> root_ = nullptr; // root node
  std::vector<Vec3f> cloud_;                     // input point cloud
  std::map<int, OctoTreeNode *> nodes_;          // for bookkeeping
  size_t size_ = 0;                              // number of leaf nodes
  int tree_node_id_ = 0;                         // id allocator for tree nodes

  // flann
  bool approximate_ = false;
  float alpha_ = 1.0; // ANN distance multiplier
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_OCTO_TREE_H
