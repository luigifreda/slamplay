#include <Mahi/Gui.hpp>

// Define a macro to create a version number in a more readable format
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if GCC_VERSION >= 110000
#include <fmt/format.h>
namespace fmt {
namespace v6 = v8;
}
#endif
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