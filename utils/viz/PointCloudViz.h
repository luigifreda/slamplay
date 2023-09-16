#pragma once 


#include <iostream>
#include <thread>

#include <pangolin/pangolin.h>
#include <unistd.h>

template <typename PointCloud>
class PointCloudViz 
{
public:     

    ~PointCloudViz() 
    {
        stop();
        std::cout << "PointCloudViz destroyed\n";
    }
    
    void start() 
    {
        running = true; 
        t = std::thread(&PointCloudViz::show_, this);
    }

    void stop()
    {
        running = false;         
        if (t.joinable()) {
            t.join();
        }
    }

    void update(PointCloud& cloudIn)
    {
        std::lock_guard<std::mutex> guard(m);
        pointcloud = cloudIn;
    }

protected: 

    void show_() 
    {
        pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1, 0, 0, 0, 0.0, -1.0, 0.0)
        );

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
                // NOTE: a buffer object could be used 
                glBegin(GL_POINTS);
                for (auto &p: pointcloud.points) {
                    glColor3f(p.r/ 255.0, p.g/ 255.0, p.b/ 255.0);
                    glVertex3f(p.x, p.y, p.z);
                }
                glEnd();
            }

            pangolin::FinishFrame();
            usleep(5000); //sleep 5 ms
        }
        running = false; 
        return;
    }

protected: 

    PointCloud pointcloud; 
    std::mutex m; // not caring much about performances here 
    std::thread t;    
    std::atomic_bool running{false};
};