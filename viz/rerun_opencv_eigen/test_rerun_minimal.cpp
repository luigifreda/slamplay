#include <rerun.hpp>
#include <rerun/demo_utils.hpp>

#include <chrono>
#include <random>
#include <thread>

using rerun::demo::grid3d;

int main() {
    std::random_device rd;

    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const auto rec = rerun::RecordingStream("rerun_minimal");
    // Try to spawn a new viewer instance.
    rec.spawn().exit_on_failure();

    while (true) {
        // Create some data using the `grid` utility function.
        std::vector<rerun::Position3D> points = grid3d<rerun::Position3D, float>(-10.f, 10.f, 10);
        std::vector<rerun::Color> colors = grid3d<rerun::Color, uint8_t>(0, 255, 10);

        // generate random radius
        float radius = std::uniform_real_distribution<float>(0.1f, 1.0f)(rd);

        // Log the "my_points" entity with our data, using the `Points3D` archetype.
        rec.log("my_points", rerun::Points3D(points).with_colors(colors).with_radii({radius}));

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}
