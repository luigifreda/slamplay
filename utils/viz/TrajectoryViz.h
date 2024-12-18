#pragma once

#include "sophus/se3.hpp"
#include "viz/PangolinUtils.h"

#include <iostream>
#include <thread>

#include <pangolin/pangolin.h>
#include <unistd.h>

namespace slamplay {

class TrajectoryViz {
   public:
    ~TrajectoryViz() {
        stop();
        std::cout << "TrajectoryViz destroyed\n";
    }

    void start() {
        running = true;
        t = std::thread(&TrajectoryViz::show_, this);
    }

    void stop() {
        running = false;
        if (t.joinable()) {
            t.join();
        }
    }

    template <typename Container>
    void setTrajectory(const Container &trajectoryIn) {
        static_assert(std::is_same<typename Container::value_type, Sophus::SE3d>::value,
                      "Container must hold Sophus::SE3d elements");

        std::lock_guard<std::mutex> guard(m);
        trajectory.clear();
        trajectory.insert(trajectory.end(), trajectoryIn.begin(), trajectoryIn.end());
    }

    void addPose(const Sophus::SE3d &pose) {
        std::lock_guard<std::mutex> guard(m);
        trajectory.push_back(pose);
    }

    void setDownsampleCameraVizFactor(int factor) {
        downsampleCameraVizFactor = factor;
    }

   protected:
    void show_() {
        pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1, 0, 0, 0, 0.0, -1.0, 0.0));

        pangolin::View &d_cam = pangolin::CreateDisplay()
                                    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

        while (running && pangolin::ShouldQuit() == false)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            d_cam.Activate(s_cam);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);

            {
                std::lock_guard<std::mutex> guard(m);
                for (size_t i = 0; i < trajectory.size(); i++) {
                    if (i % downsampleCameraVizFactor != 0) continue;
                    Sophus::SE3d pose = trajectory[i];
                    slamplay::render_camera(pose.matrix());
                }

                for (size_t i = 0; i < trajectory.size() - 1; i++) {
                    Eigen::Vector3d p1(trajectory[i].translation());
                    Eigen::Vector3d p2(trajectory[i + 1].translation());
                    glBegin(GL_LINES);
                    glColor3f(0.0, 0.0, 1.0);
                    glVertex3d(p1[0], p1[1], p1[2]);
                    glVertex3d(p2[0], p2[1], p2[2]);
                    glEnd();
                }
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        running = false;
        return;
    }

   protected:
    std::vector<Sophus::SE3d> trajectory;
    std::mutex m;  // not caring much about performances here
    std::thread t;
    std::atomic_bool running{false};
    int downsampleCameraVizFactor = 1;
};

}  // namespace slamplay