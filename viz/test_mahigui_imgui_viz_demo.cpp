#include <Mahi/Gui.hpp>
#include <Mahi/Util.hpp>

using namespace mahi::gui;
using namespace mahi::util;

// Inherit from Application
class ImGuiDemo : public Application {
public:
    ImGuiDemo() : Application() { }
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