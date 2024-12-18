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

#include <iostream>
#include <thread>

#include <pangolin/pangolin.h>
#include <unistd.h>

namespace slamplay {

template <typename PointCloud>
class PointCloudViz {
   public:
    ~PointCloudViz() {
        stop();
        std::cout << "PointCloudViz destroyed\n";
    }

    void start() {
        running = true;
        t = std::thread(&PointCloudViz::show_, this);
    }

    void stop() {
        running = false;
        if (t.joinable()) {
            t.join();
        }
    }

    void update(PointCloud &cloudIn) {
        std::lock_guard<std::mutex> guard(m);
        pointcloud = cloudIn;
    }

   protected:
    void show_() {
        pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
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

            {
                std::lock_guard<std::mutex> guard(m);
                glPointSize(2);
                // NOTE: a buffer object could be used here (this is just a very simple example)
                glBegin(GL_POINTS);
                for (auto &p : pointcloud.points) {
                    glColor3f(p.r / 255.0, p.g / 255.0, p.b / 255.0);
                    glVertex3f(p.x, p.y, p.z);
                }
                glEnd();
            }

            pangolin::FinishFrame();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        running = false;
        return;
    }

   protected:
    PointCloud pointcloud;
    std::mutex m;  // not caring much about performances here
    std::thread t;
    std::atomic_bool running{false};
};

}  // namespace slamplay