
#pragma once

#include "datasets/DatasetIo.h"
#include "io/filesystem.h"

#include <opencv2/highgui/highgui.hpp>

namespace slamplay {

class ReplicaDataset : public VioDataset {
    constexpr static double Ts = 1.0 / 30.0;

    size_t num_cams;

    std::string path;

    std::vector<int64_t> image_timestamps;
    std::unordered_map<int64_t, std::string> image_path;

    std::vector<int64_t> depth_timestamps;
    std::unordered_map<int64_t, std::string> depth_path;

    // vector of images for every timestamp
    // assumes vectors size is num_cams for every timestamp with null pointers for
    // missing frames
    // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

    Eigen::aligned_vector<AccelData> accel_data;
    Eigen::aligned_vector<GyroData> gyro_data;

    std::vector<int64_t> gt_timestamps;                // ordered gt timestamps
    Eigen::aligned_vector<Sophus::SE3d> gt_pose_data;  // TODO: change to eigen aligned

    int64_t mocap_to_imu_offset_ns;

   public:
    ~ReplicaDataset() {};

    size_t get_num_cams() const { return num_cams; }

    std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }
    std::vector<int64_t> &get_depth_timestamps() { return depth_timestamps; }

    const Eigen::aligned_vector<AccelData> &get_accel_data() const {
        return accel_data;
    }
    const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
        return gyro_data;
    }
    const std::vector<int64_t> &get_gt_timestamps() const {
        return gt_timestamps;
    }
    const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
        return gt_pose_data;
    }
    int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

    std::vector<ImageData> get_image_data(int64_t t_ns) {
        std::vector<ImageData> res(num_cams);
        const std::vector<std::string> folder = {"/results/"};

        for (size_t i = 0; i < num_cams; i++) {
            std::string full_image_path = path + folder[0] + image_path[t_ns];

            if (fs::exists(full_image_path)) {
                cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);
                if (!img.empty()) {
                    res[i].img = img;
                } else {
                    std::cerr << "Failed to load RGB image: " << full_image_path << std::endl;
                }
            }
        }
        return res;
    }

    std::vector<DepthData> get_depth_data(int64_t t_ns) {
        std::vector<DepthData> res(num_cams);
        const std::vector<std::string> folder = {"/results/"};

        for (size_t i = 0; i < num_cams; i++) {
            std::string full_depth_path = path + folder[0] + depth_path[t_ns];

            if (fs::exists(full_depth_path)) {
                cv::Mat depth_img = cv::imread(full_depth_path, cv::IMREAD_UNCHANGED);
                if (!depth_img.empty()) {
                    DepthData depth_data;
                    depth_data.img = depth_img;
                    res[i] = depth_data;
                } else {
                    std::cerr << "Failed to load depth image at " << full_depth_path << std::endl;
                }
            } else {
                std::cerr << "Depth image not found: " << full_depth_path << std::endl;
            }
        }

        return res;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    friend class ReplicaIO;
};

class ReplicaIO : public DatasetIoInterface {
   public:
    ReplicaIO() {}

    void read(const std::string &path) {
        if (!fs::exists(path))
            std::cerr << "No dataset found in " << path << std::endl;

        data.reset(new ReplicaDataset);

        data->num_cams = 1;
        data->path = path;

        if (fs::exists(path + "/results")) {
            read_image_and_depth_paths(path);
        } else {
            std::cerr << "No results found in " << path << std::endl;
        }

        if (fs::exists(path + "/traj.txt")) {
            read_gt_data_pose(path + "/traj.txt");
        } else {
            std::cerr << "No ground truth data found in " << path << std::endl;
        }
    }

    void reset() { data.reset(); }

    VioDatasetPtr get_data() { return data; }

   private:
    std::string generate_filename(size_t number) {
        std::ostringstream oss;
        oss << std::setw(6) << std::setfill('0') << number;  // 6-digit zero-padded integer
        return oss.str();
    }

    Eigen::Matrix3d enforceRotationMatrix(const Eigen::Matrix3d &mat) {
        // Perform Singular Value Decomposition (SVD)
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(mat, Eigen::ComputeFullU | Eigen::ComputeFullV);

        // Correct the rotation matrix
        Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();

        // Ensure determinant is +1 (proper rotation)
        if (R.determinant() < 0) {
            Eigen::Matrix3d U = svd.matrixU();
            U.col(2) *= -1;  // Flip the sign of the last column
            R = U * svd.matrixV().transpose();
        }

        return R;
    }

    void read_image_and_depth_paths(const std::string &path) {
        const std::string full_folder_path = path + "/results";

        std::vector<std::string> color_paths;
        std::vector<std::string> depth_paths;

        // number of files in the path
        size_t num_depth_files = 0;
        if (fs::exists(full_folder_path) && fs::is_directory(full_folder_path)) {
            for (const auto &entry : fs::directory_iterator(full_folder_path)) {
                if (fs::is_regular_file(entry) && entry.path().extension() == ".png") {
                    num_depth_files++;
                }
            }
        } else {
            std::cerr << "Invalid folder path: " << full_folder_path << std::endl;
        }

        data->image_timestamps.resize(num_depth_files);
        data->depth_timestamps.resize(num_depth_files);
        for (size_t i = 0; i < num_depth_files; i++) {
            const int64_t timestamp = static_cast<int64_t>(double(i) * data->Ts * 1e9);
            data->image_timestamps[i] = timestamp;
            data->depth_timestamps[i] = timestamp;
            const std::string image_name = "frame" + generate_filename(i) + ".jpg";
            const std::string depth_name = "depth" + generate_filename(i) + ".png";
            data->image_path[timestamp] = image_name;
            data->depth_path[timestamp] = depth_name;
        }
    }

    void read_gt_data_pose(const std::string &filepath) {
        std::ifstream f(filepath);
        if (!f.is_open()) {
            std::cerr << "Failed to open traj.txt at " << filepath << std::endl;
            return;
        }

        std::string line;
        int count = 0;
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::vector<double> matrix_data(16);
            for (size_t i = 0; i < 16; ++i) {
                ss >> matrix_data[i];
            }

            Eigen::Matrix4d pose_matrix;
            for (size_t i = 0; i < 4; ++i) {
                for (size_t j = 0; j < 4; ++j) {
                    pose_matrix(i, j) = matrix_data[i * 4 + j];
                }
            }

            // Invert the pose as done in the Python code
            pose_matrix = pose_matrix.inverse();

            Sophus::SO3d rot(enforceRotationMatrix(pose_matrix.block<3, 3>(0, 0)));
            Eigen::Vector3d t = pose_matrix.block<3, 1>(0, 3);

            Sophus::SE3d pose(rot, t);
            const auto &t_ns = data->image_timestamps[count++];
            data->gt_timestamps.emplace_back(t_ns);
            data->gt_pose_data.emplace_back(Eigen::Quaterniond(pose.rotationMatrix()), pose.translation());
        }
    }

    std::shared_ptr<ReplicaDataset> data;
};

}  // namespace slamplay
