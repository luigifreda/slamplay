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