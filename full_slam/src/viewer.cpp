//
// Created by gaoxiang on 19-5-4. 
// From https://github.com/gaoxiang12/slambook2
// Modified by Luigi Freda later for slamplay 
//

#include "myslam/viewer.h"
#include "myslam/feature.h"
#include "myslam/frame.h"
#include "myslam/config.h"

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>

namespace myslam {

Viewer::Viewer() {
    viewer_thread_ = std::thread(std::bind(&Viewer::ThreadLoop, this));

    if(Config::IsAvailable("Camera.scale")){
        camera_scale_ = Config::Get<double>("Camera.scale");
    }
}

void Viewer::Close() {
    viewer_running_ = false;
    viewer_thread_.join();
}

void Viewer::AddCurrentFrame(Frame::Ptr current_frame) {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    current_frame_ = current_frame;
}

void Viewer::UpdateMap() {
    std::unique_lock<std::mutex> lck(viewer_data_mutex_);
    assert(map_ != nullptr);
    
    active_keyframes_ = map_->GetActiveKeyFrames();
    all_keyframes_ = map_->GetAllKeyFrames();    
    for(auto it=all_keyframes_.begin(); it!=all_keyframes_.end();) {
        if(active_keyframes_.find(it->first)!=active_keyframes_.end()) {
            it = all_keyframes_.erase(it);
        }
        else {
            it++;
        }
    }

    active_landmarks_ = map_->GetActiveMapPoints();
    all_landmarks_ = map_->GetAllMapPoints();
    for(auto it=all_landmarks_.begin(); it!=all_landmarks_.end();) {
        if(active_landmarks_.find(it->first)!=active_landmarks_.end()) {
            it = all_landmarks_.erase(it);
        }
        else {
            it++;
        }
    }    

    map_updated_ = true;
}

void Viewer::ThreadLoop() {
#if 0    
    constexpr int window_with = 1024; 
    constexpr int window_height = 768; 
#else 
    constexpr int window_with = 1600; 
    constexpr int window_height = 1200; 
#endif     

    pangolin::CreateWindowAndBind("MySLAM", window_with, window_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState vis_camera(
        pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -10, -20, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& vis_display =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera));

    const float blue[3] = {0, 0, 1};
    const float green[3] = {0, 1, 0};

    while (!pangolin::ShouldQuit() && viewer_running_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        vis_display.Activate(vis_camera);

        {
            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            if (current_frame_) {
                DrawFrame(current_frame_, green);
                FollowCurrentFrame(vis_camera);

                cv::Mat img = PlotFrameImage();
                cv::imshow("image", img);
            }

            if (map_) {
                DrawAllMapPoints();                
                DrawAllKeyFrames();      
                DrawTrajectory();                      

                DrawActiveMapPoints();                
                DrawActiveKeyFrames();                  
            }
        }

        cv::waitKey(1);

        pangolin::FinishFrame();
        usleep(5000);
    }

    LOG(INFO) << "Stop viewer";
}

cv::Mat Viewer::PlotFrameImage() {
    cv::Mat img_out;
    cv::cvtColor(current_frame_->left_img_, img_out, cv::COLOR_GRAY2BGR);
    for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
        if (current_frame_->features_left_[i]->map_point_.lock()) {
            auto feat = current_frame_->features_left_[i];
            cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0), 2);
        }
    }
    return img_out;
}

void Viewer::FollowCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    SE3 Twc = current_frame_->Pose().inverse();
    pangolin::OpenGlMatrix m(Twc.matrix());
    vis_camera.Follow(m, true);
}

void Viewer::DrawFrame(Frame::Ptr frame, const float* color) {
    SE3 Twc = frame->Pose().inverse();
    const float sz = camera_scale_;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();

    Sophus::Matrix4f m = Twc.matrix().template cast<float>();
    glMultMatrixf((GLfloat*)m.data());

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}

void Viewer::DrawAllKeyFrames()
{
    constexpr float green[3] = {0, 1.0, 0};      
    DrawFrames(all_keyframes_, green);
}


void Viewer::DrawActiveKeyFrames()
{
    constexpr float red[3] = {1.0, 0, 0};   
    DrawFrames(active_keyframes_, red);    
}

void Viewer::DrawFrames(const std::unordered_map<unsigned long, Frame::Ptr>& keyframes, 
                        const float color[3]) {
    for (auto& kf : keyframes) {
        DrawFrame(kf.second, color);
    }   
}


void Viewer::DrawTrajectory() {
    constexpr float color[3] = {1, 0, 0};        
    glLineWidth(2);     
    glBegin(GL_LINE_STRIP);    
    glColor3f(color[0], color[1], color[2]);        
    for (unsigned long id=0; id<=Frame::factory_id; id++) {
        auto it = all_keyframes_.find(id);
        if(it!=all_keyframes_.end())
        {
            const auto pos = it->second->Pose().inverse().translation();               
            glVertex3d(pos[0], pos[1], pos[2]);        
        }
    }    
    glEnd();    
}

void Viewer::DrawActiveMapPoints()
{
    constexpr float red[3] = {1.0, 0, 0};    
    DrawPoints(active_landmarks_, red, 4);
}

void Viewer::DrawAllMapPoints()
{
    constexpr float blue[3] = {0, 0, 1.0};    
    DrawPoints(all_landmarks_, blue, 3);
}

void Viewer::DrawPoints(const std::unordered_map<unsigned long, MapPoint::Ptr>& landmarks, 
                        const float color[3], GLfloat point_size) {
    glPointSize(point_size);
    glBegin(GL_POINTS);
    glColor3f(color[0], color[1], color[2]);    
    for (auto& landmark : landmarks) {
        const auto pos = landmark.second->Pos();
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

}  // namespace myslam
