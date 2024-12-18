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
#include <Mahi/Gui.hpp>

#include <fmt_adapt.h>
#include <Mahi/Util.hpp>
#include <cmath>

using namespace mahi::gui;
using namespace mahi::util;

void DemoImplotSimple(bool* p_open) {
    // App logic and/or ImGui code goes here
    ImGui::Begin("ImPlot Demo", p_open, ImGuiWindowFlags_MenuBar);

    const double pi = 3.1415;
    const double spanx = 2 * pi;
    const int numSamples = 1000;
    static std::vector<double> x, y1, y2;
    if (x.empty())
    {
        const double delta = spanx / numSamples;
        for (int i = 0; i < numSamples; ++i)
        {
            double x_ = delta * i;
            x.push_back(x_);
            y1.push_back(cos(x_));
            y2.push_back(sin(x_));
        }
    }

    ImPlot::SetNextPlotLimits(0, spanx, -2.0, 2.0);
    if (ImPlot::BeginPlot("Plot", NULL, NULL, ImVec2(-1, 700)))
    {
        ImPlot::PlotLine("y1", x.data(), y1.data(), x.size());
        ImPlot::PlotLine("y2", x.data(), y2.data(), x.size());
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
        DemoImplotSimple(&p_open);
        if (!p_open)
            quit();
    }
};

int main(int, char*[]) {
    MyApp app;
    app.run();
    return 0;
}