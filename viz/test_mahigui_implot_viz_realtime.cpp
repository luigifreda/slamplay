#include <Mahi/Gui.hpp>

#include <fmt_adapt.h>
#include <Mahi/Util.hpp>
#include <cmath>

using namespace mahi::gui;
using namespace mahi::util;

// utility structure for realtime plot
struct ScrollingBuffer {
    int MaxSize;
    int Offset;
    ImVector<ImVec2> Data;
    ScrollingBuffer(int max_size = 2000) {
        MaxSize = max_size;
        Offset = 0;
        Data.reserve(MaxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize)
            Data.push_back(ImVec2(x, y));
        else {
            Data[Offset] = ImVec2(x, y);
            Offset = (Offset + 1) % MaxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            Offset = 0;
        }
    }
};

// utility structure for realtime plot
struct RollingBuffer {
    float Span;
    ImVector<ImVec2> Data;
    RollingBuffer() {
        Span = 10.0f;
        Data.reserve(2000);
    }
    void AddPoint(float x, float y) {
        float xmod = fmodf(x, Span);
        if (!Data.empty() && xmod < Data.back().x)
            Data.shrink(0);
        Data.push_back(ImVec2(xmod, y));
    }
};

void DemoImplotMouse(bool* p_open) {
    // ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
    // ImGui::SetNextWindowSize(ImVec2(600, 750), ImGuiCond_FirstUseEver);
    ImGui::Begin("ImPlot Demo", p_open, ImGuiWindowFlags_MenuBar);

    ImGui::BulletText("Move your mouse to change the data!");
    ImGui::BulletText("This example assumes 60 FPS. Higher FPS requires larger buffer size.");
    static ScrollingBuffer sdata1, sdata2;
    static RollingBuffer rdata1, rdata2;
    ImVec2 mouse = ImGui::GetMousePos();
    static float t = 0;
    t += ImGui::GetIO().DeltaTime;
    sdata1.AddPoint(t, mouse.x * 0.0005f);
    rdata1.AddPoint(t, mouse.x * 0.0005f);
    sdata2.AddPoint(t, mouse.y * 0.0005f);
    rdata2.AddPoint(t, mouse.y * 0.0005f);

    static float history = 10.0f;
    ImGui::SliderFloat("History", &history, 1, 30, "%.1f s");
    rdata1.Span = history;
    rdata2.Span = history;

    static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
    // static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_Time;
    ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##Scrolling", NULL, NULL, ImVec2(-1, 300), 0, rt_axis, rt_axis | ImPlotAxisFlags_LockMin)) {
        ImPlot::PlotShaded("Data 1", &sdata1.Data[0].x, &sdata1.Data[0].y, sdata1.Data.size(), 0, sdata1.Offset, 2 * sizeof(float));
        ImPlot::PlotLine("Data 2", &sdata2.Data[0].x, &sdata2.Data[0].y, sdata2.Data.size(), sdata2.Offset, 2 * sizeof(float));
        ImPlot::EndPlot();
    }
    ImPlot::SetNextPlotLimitsX(0, history, ImGuiCond_Always);
    if (ImPlot::BeginPlot("##Rolling", NULL, NULL, ImVec2(-1, 300), 0, rt_axis, rt_axis)) {
        ImPlot::PlotLine("Data 1", &rdata1.Data[0].x, &rdata1.Data[0].y, rdata1.Data.size(), 0, 2 * sizeof(float));
        ImPlot::PlotLine("Data 2", &rdata2.Data[0].x, &rdata2.Data[0].y, rdata2.Data.size(), 0, 2 * sizeof(float));
        ImPlot::EndPlot();
    }

    ImGui::End();
}

// Inherit from Application
class MyApp : public Application {
   public:
    MyApp() : Application() {
        ImGui::StyleColorsMahiDark3();
    }
    // Override update (called once per frame)
    void update() override {
        static bool p_open = true;
        DemoImplotMouse(&p_open);
        if (!p_open)
            quit();
    }
};

int main(int, char*[]) {
    MyApp app;
    app.run();
    return 0;
}