#ifndef SLAM_IN_AUTO_DRIVING_KDTREE_H
#define SLAM_IN_AUTO_DRIVING_KDTREE_H

#include "ad/common/eigen_types.h"
#include "ad/pointcloud/point_types.h"

#include <glog/logging.h>
#include <map>
#include <queue>

namespace sad {

/// Kd tree node, binary tree structure; uses raw pointers internally, exposes a
/// shared_ptr for root
struct KdTreeNode {
  int id_ = -1;
  int point_idx_ = 0;           // point index
  int axis_index_ = 0;          // split axis
  float split_thresh_ = 0.0;    // split threshold
  KdTreeNode *left_ = nullptr;  // left subtree
  KdTreeNode *right_ = nullptr; // right subtree

  bool IsLeaf() const {
    return left_ == nullptr && right_ == nullptr;
  } // whether it is a leaf node
};

/// Used to store knn results
struct NodeAndDistance {
  NodeAndDistance(KdTreeNode *node, float dis2)
      : node_(node), distance2_(dis2) {}
  KdTreeNode *node_ = nullptr;
  float distance2_ = 0; // squared distance, used for comparison

  bool operator<(const NodeAndDistance &other) const {
    return distance2_ < other.distance2_;
  }
};

/**
 * Hand-written kd tree
 * Test the recall of this kd tree!
 */
class KdTree {
public:
  explicit KdTree() = default;
  ~KdTree() { Clear(); }

  bool BuildTree(const CloudPtr &cloud);

  /// Get k nearest neighbors
  bool GetClosestPoint(const PointType &pt, std::vector<int> &closest_idx,
                       int k = 5);

  /// Find nearest neighbors for point cloud in parallel
  bool GetClosestPointMT(const CloudPtr &cloud,
                         std::vector<std::pair<size_t, size_t>> &matches,
                         int k = 5);

  /// Set the ANN distance multiplier
  void SetEnableANN(bool use_ann = true, float alpha = 0.1) {
    approximate_ = use_ann;
    alpha_ = alpha;
  }

  /// Return the number of nodes
  size_t size() const { return size_; }

  /// Clear data
  void Clear();

  /// Print all node information
  void PrintAll();

private:
  /// kdtree construction
  /**
   * Insert points at node
   * @param points
   * @param node
   */
  void Insert(const IndexVec &points, KdTreeNode *node);

  /**
   * Compute the splitting plane for a point set
   * @param points input point cloud
   * @param axis   split axis
   * @param th     threshold
   * @param left   left subset
   * @param right  right subset
   * @return
   */
  bool FindSplitAxisAndThresh(const IndexVec &point_idx, int &axis, float &th,
                              IndexVec &left, IndexVec &right);

  void Reset();

  /// Squared distance between two points
  static inline float Dis2(const Vec3f &p1, const Vec3f &p2) {
    return (p1 - p2).squaredNorm();
  }

  // Knn related
  /**
   * Check knn for the given point on a kdtree node, called recursively
   * @param pt     query point
   * @param node   kdtree node
   */
  void Knn(const Vec3f &pt, KdTreeNode *node,
           std::priority_queue<NodeAndDistance> &result) const;

  /**
   * For a leaf node, compute its distance to the query point and try to insert
   * into results
   * @param pt    query point
   * @param node  Kdtree node
   */
  void ComputeDisForLeaf(const Vec3f &pt, KdTreeNode *node,
                         std::priority_queue<NodeAndDistance> &result) const;

  /**
   * Check whether the subtree under node needs to be expanded
   * @param pt   query point
   * @param node Kdtree node
   * @return true if expansion is needed
   */
  bool NeedExpand(const Vec3f &pt, KdTreeNode *node,
                  std::priority_queue<NodeAndDistance> &knn_result) const;

  int k_ = 5;                                   // knn neighbor count
  std::shared_ptr<KdTreeNode> root_ = nullptr;  // root node
  std::vector<Vec3f> cloud_;                    // input point cloud
  std::unordered_map<int, KdTreeNode *> nodes_; // for bookkeeping

  size_t size_ = 0;      // number of leaf nodes
  int tree_node_id_ = 0; // id allocator for kdtree nodes

  // approximate nearest neighbor
  bool approximate_ = true;
  float alpha_ = 0.1;
};

} // namespace sad

#endif // SLAM_IN_AUTO_DRIVING_KDTREE_H
