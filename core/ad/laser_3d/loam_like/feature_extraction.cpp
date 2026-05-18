#include "ad/laser_3d/loam_like/feature_extraction.h"

#include <glog/logging.h>

namespace sad {

void FeatureExtraction::Extract(FullCloudPtr pc_in, CloudPtr pc_out_edge,
                                CloudPtr pc_out_surf) {
  int num_scans = 16;
  std::vector<CloudPtr> scans_in_each_line; // point clouds split by scan line
  for (int i = 0; i < num_scans; i++) {
    scans_in_each_line.emplace_back(new PointCloudType);
  }

  for (auto &pt : pc_in->points) {
    assert(pt.ring >= 0 && pt.ring < num_scans);
    PointType p;
    p.x = pt.x;
    p.y = pt.y;
    p.z = pt.z;
    p.intensity = pt.intensity;

    scans_in_each_line[pt.ring]->points.emplace_back(p);
  }

  // Compute curvature
  for (int i = 0; i < num_scans; i++) {
    if (scans_in_each_line[i]->points.size() < 131) {
      continue;
    }

    std::vector<IdAndValue> cloud_curvature; // curvature for each scan line
    int total_points = scans_in_each_line[i]->points.size() - 10;
    for (int j = 5; j < (int)scans_in_each_line[i]->points.size() - 5; j++) {
      // Leave margin at both ends, sample 10 surrounding points for averaging
      double diffX = scans_in_each_line[i]->points[j - 5].x +
                     scans_in_each_line[i]->points[j - 4].x +
                     scans_in_each_line[i]->points[j - 3].x +
                     scans_in_each_line[i]->points[j - 2].x +
                     scans_in_each_line[i]->points[j - 1].x -
                     10 * scans_in_each_line[i]->points[j].x +
                     scans_in_each_line[i]->points[j + 1].x +
                     scans_in_each_line[i]->points[j + 2].x +
                     scans_in_each_line[i]->points[j + 3].x +
                     scans_in_each_line[i]->points[j + 4].x +
                     scans_in_each_line[i]->points[j + 5].x;
      double diffY = scans_in_each_line[i]->points[j - 5].y +
                     scans_in_each_line[i]->points[j - 4].y +
                     scans_in_each_line[i]->points[j - 3].y +
                     scans_in_each_line[i]->points[j - 2].y +
                     scans_in_each_line[i]->points[j - 1].y -
                     10 * scans_in_each_line[i]->points[j].y +
                     scans_in_each_line[i]->points[j + 1].y +
                     scans_in_each_line[i]->points[j + 2].y +
                     scans_in_each_line[i]->points[j + 3].y +
                     scans_in_each_line[i]->points[j + 4].y +
                     scans_in_each_line[i]->points[j + 5].y;
      double diffZ = scans_in_each_line[i]->points[j - 5].z +
                     scans_in_each_line[i]->points[j - 4].z +
                     scans_in_each_line[i]->points[j - 3].z +
                     scans_in_each_line[i]->points[j - 2].z +
                     scans_in_each_line[i]->points[j - 1].z -
                     10 * scans_in_each_line[i]->points[j].z +
                     scans_in_each_line[i]->points[j + 1].z +
                     scans_in_each_line[i]->points[j + 2].z +
                     scans_in_each_line[i]->points[j + 3].z +
                     scans_in_each_line[i]->points[j + 4].z +
                     scans_in_each_line[i]->points[j + 5].z;
      IdAndValue distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
      cloud_curvature.push_back(distance);
    }

    // Extract features from each sector, dividing 360 degrees into 6 sectors
    for (int j = 0; j < 6; j++) {
      int sector_length = (int)(total_points / 6);
      int sector_start = sector_length * j;
      int sector_end = sector_length * (j + 1) - 1;
      if (j == 5) {
        sector_end = total_points - 1;
      }

      std::vector<IdAndValue> sub_cloud_curvature(
          cloud_curvature.begin() + sector_start,
          cloud_curvature.begin() + sector_end);

      ExtractFromSector(scans_in_each_line[i], sub_cloud_curvature, pc_out_edge,
                        pc_out_surf);
    }
  }
}

void FeatureExtraction::ExtractFromSector(
    const CloudPtr &pc_in, std::vector<IdAndValue> &cloud_curvature,
    CloudPtr &pc_out_edge, CloudPtr &pc_out_surf) {
  // Sort by curvature
  std::sort(cloud_curvature.begin(), cloud_curvature.end(),
            [](const IdAndValue &a, const IdAndValue &b) {
              return a.value_ < b.value_;
            });

  int largest_picked_num = 0;
  int point_info_count = 0;

  /// Search starting from the highest curvature, select edge points with the
  /// largest curvature
  std::vector<int> picked_points; // marks selected edge points; nearby points
                                  // will not be selected
  for (int i = cloud_curvature.size() - 1; i >= 0; i--) {
    int ind = cloud_curvature[i].id_;
    if (std::find(picked_points.begin(), picked_points.end(), ind) ==
        picked_points.end()) {
      if (cloud_curvature[i].value_ <= 0.1) {
        break;
      }

      largest_picked_num++;
      picked_points.push_back(ind);

      if (largest_picked_num <= 20) {
        pc_out_edge->push_back(pc_in->points[ind]);
        point_info_count++;
      } else {
        break;
      }

      for (int k = 1; k <= 5; k++) {
        double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
        double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
        double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
          break;
        }
        picked_points.push_back(ind + k);
      }
      for (int k = -1; k >= -5; k--) {
        double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
        double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
        double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
          break;
        }
        picked_points.push_back(ind + k);
      }
    }
  }

  /// Select planar points with small curvature
  for (int i = 0; i <= (int)cloud_curvature.size() - 1; i++) {
    int ind = cloud_curvature[i].id_;
    if (std::find(picked_points.begin(), picked_points.end(), ind) ==
        picked_points.end()) {
      pc_out_surf->push_back(pc_in->points[ind]);
    }
  }
}

} // namespace sad