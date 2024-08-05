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

// Inherit from Application
class ImGuiDemo : public Application {
   public:
    ImGuiDemo() : Application() {}
    void update() override {
        // Official ImGui demo (see imgui_demo.cpp for full example)
        static bool open = true;
        ImGui::ShowDemoWindow(&open);
        if (!open)
            quit();
    }
};

int main() {
    ImGuiDemo app;
    app.run();
    return 0;
}