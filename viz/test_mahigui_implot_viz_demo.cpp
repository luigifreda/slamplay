#include <Mahi/Gui.hpp>

#include <fmt_adapt.h>
#include <Mahi/Util.hpp>

using namespace mahi::gui;
using namespace mahi::util;

class ImPlotDemo : public Application {
   public:
    ImPlotDemo() : Application() {
        ImGui::StyleColorsMahiDark3();
    }
    void update() override {
        static bool p_open = true;
        ImPlot::ShowDemoWindow(&p_open);
        if (!p_open)
            quit();
    }
};

int main(int argc, char const *argv[]) {
    ImPlotDemo demo;
    demo.run();
    return 0;
}