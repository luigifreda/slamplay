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
