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
#pragma once

#include "sophus/se3.hpp"
#include "viz/PangolinUtils.h"

#include <iostream>
#include <thread>

#include <pangolin/pangolin.h>
#include <unistd.h>

namespace slamplay {

class Trajectory {
   public:
    using Ptr = std::shared_ptr<Trajectory>;
    using Color = Eigen::Vector3f;

   public:
    Trajectory(const std::vector<Sophus::SE3d> &trajectoryIn, const Color &colorIn, const std::string &nameIn) : trajectory(trajectoryIn), name(nameIn) {
        if (colorIn.array().maxCoeff() > 1.0) {
            color = colorIn / 255.0;
        } else {
            color = colorIn;
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

    void setColor(const Color &colorIn) {
        std::lock_guard<std::mutex> guard(m);
        if (colorIn.array().maxCoeff() > 1.0) {
            color = colorIn / 255.0;
        } else {
            color = colorIn;
        }
    }

    std::mutex m;
    std::vector<Sophus::SE3d> trajectory;
    Color color;
    std::string name;
};

// Viz for visualizing trajectories
class TrajectoryViz {
   public:
    constexpr static int kSleepTimeMs = 20;

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

    bool isRunning() { return running; }

    template <typename Container>
    void addTrajectory(const Container &trajectoryIn, const Trajectory::Color &colorIn, const std::string &name) {
        static_assert(std::is_same<typename Container::value_type, Sophus::SE3d>::value,
                      "Container must hold Sophus::SE3d elements");

        std::lock_guard<std::mutex> guard(m);
        trajectories[name] = std::make_shared<Trajectory>(trajectoryIn, colorIn, name);
    }

    template <typename Container>
    void setTrajectory(const Container &trajectoryIn, const Trajectory::Color &colorIn, const std::string &name) {
        static_assert(std::is_same<typename Container::value_type, Sophus::SE3d>::value,
                      "Container must hold Sophus::SE3d elements");

        std::lock_guard<std::mutex> guard(m);
        auto it = trajectories.find(name);
        if (it != trajectories.end()) {
            it->second->setTrajectory(trajectoryIn);
        } else {
            std::cerr << "TrajectoryViz::setTrajectory: trajectory " << name << " not found" << std::endl;
        }
    }

    void addPose(const Sophus::SE3d &pose, const std::string &name) {
        std::lock_guard<std::mutex> guard(m);
        auto it = trajectories.find(name);
        if (it != trajectories.end()) {
            it->second->addPose(pose);
        } else {
            std::cerr << "TrajectoryViz::addPose: trajectory " << name << " not found" << std::endl;
        }
    }

    void setDownsampleCameraVizFactor(int factor) {
        downsampleCameraVizFactor = factor;
    }

    Trajectory::Ptr getTrajectory(const std::string &name) {
        std::lock_guard<std::mutex> guard(m);
        auto it = trajectories.find(name);
        if (it == trajectories.end()) {
            std::cerr << "TrajectoryViz::getTrajectory: trajectory " << name << " not found" << std::endl;
            return nullptr;
        } else {
            return it->second;
        }
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

            // first get all the keys
            std::vector<std::string> keys;
            {
                std::lock_guard<std::mutex> guard(m);
                for (auto it = trajectories.begin(); it != trajectories.end(); it++) {
                    keys.push_back(it->first);
                }
            }

            // draw all the trajectories one at time without blocking the full map for the rendering
            Trajectory::Ptr traj;
            for (std::string key : keys) {
                traj = getTrajectory(key);
                if (!traj) continue;

                std::lock_guard<std::mutex> guard(traj->m);
                const auto &trajectory = traj->trajectory;
                for (size_t i = 0; i < trajectory.size(); i++) {
                    if (i % downsampleCameraVizFactor != 0) continue;
                    Sophus::SE3d pose = trajectory[i];
                    slamplay::renderCamera(pose.matrix(), traj->color);
                }

                glColor3f(traj->color[0], traj->color[1], traj->color[2]);
                for (size_t i = 0; i < trajectory.size() - 1; i++) {
                    Eigen::Vector3d p1(trajectory[i].translation());
                    Eigen::Vector3d p2(trajectory[i + 1].translation());
                    glBegin(GL_LINES);
                    glVertex3d(p1[0], p1[1], p1[2]);
                    glVertex3d(p2[0], p2[1], p2[2]);
                    glEnd();
                }
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(kSleepTimeMs));
        }
        running = false;
        return;
    }

   protected:
    std::map<std::string, Trajectory::Ptr> trajectories;
    std::mutex m;
    std::thread t;
    std::atomic_bool running{false};
    int downsampleCameraVizFactor = 1;
};

}  // namespace slamplay