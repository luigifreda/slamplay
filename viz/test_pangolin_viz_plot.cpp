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
#include <pangolin/display/display.h>
#include <pangolin/plot/plotter.h>

#include <pangolin/display/default_font.h>

#include <chrono>
#include <cmath>
#include <thread>

int main(/*int argc, char* argv[]*/)
{
  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main",640,480);

  // can do this only after a pangolin context has been created 
  pangolin::set_font_size(30);

  // Data logger object
  pangolin::DataLog log;

  // Optionally add named labels
  std::vector<std::string> labels;
  labels.push_back(std::string("sin(t)"));
  labels.push_back(std::string("cos(t)"));
  labels.push_back(std::string("sin(t)+cos(t)"));
  log.SetLabels(labels);

  // OpenGL 'view' of data. We might have many views of the same data.

  const double tinc = 0.1f;    // [seconds] should be bigger than tsleep (it's the time resolution)
  const double tsleep = 0.005; // [seconds]
  const int64_t tsleep_ms = tsleep*1000; 

#define JUST_PLOT_TIME_TO_GET_AN_IDEA_OF_SCALES 0

  const float left=0.0f;
  const float right=1000.0*tinc; // set the xrange to 1000 deltaT 

  // OpenGL 'view' of data. We might have many views of the same data.
#if JUST_PLOT_TIME_TO_GET_AN_IDEA_OF_SCALES
  pangolin::Plotter plotter(&log, left,right,  0.0f,10.0f,  1.0f,1.0f);
#else
  pangolin::Plotter plotter(&log, left,right,  -2.0f,2.0f,  10.0*tinc,1.0f);
#endif   
  plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
  plotter.Track("$i");

  // Add some sample annotations to the plot
  plotter.AddMarker(pangolin::Marker::Vertical,   -1000, pangolin::Marker::LessThan, pangolin::Colour::Blue().WithAlpha(0.2f) );
  plotter.AddMarker(pangolin::Marker::Horizontal,   100, pangolin::Marker::GreaterThan, pangolin::Colour::Red().WithAlpha(0.2f) );
  plotter.AddMarker(pangolin::Marker::Horizontal,    10, pangolin::Marker::Equal, pangolin::Colour::Green().WithAlpha(0.2f) );

  pangolin::DisplayBase().AddDisplay(plotter);

  const double T = 10;           // [s]
  const double omega = 2*M_PI/T; // [rad/s]

  double t = 0;
  std::chrono::steady_clock::time_point time0 = std::chrono::steady_clock::now();  
  // Default hooks for exiting (Esc) and fullscreen (tab).
  while( !pangolin::ShouldQuit() )
  {
    std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();    
    std::chrono::duration<double> elapsed = std::chrono::duration_cast< std::chrono::duration<double >> (time - time0);
    const double elapsed_s = elapsed.count();
    if(elapsed_s>=tinc)
    {
      t += tinc; //elapsed_s;      
    }
    else
    {
      const int64_t remanining_ms = (tinc-elapsed_s) * 1000; 
      const int64_t sleep_ms = remanining_ms > tsleep_ms? tsleep_ms : remanining_ms;
      if(sleep_ms>0) std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));  
      continue; 
    } 

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

#if JUST_PLOT_TIME_TO_GET_AN_IDEA_OF_SCALES
    log.Log(t);
#else    
    log.Log(sin(omega*t), cos(omega*t), sin(omega*t)+cos(omega*t));
#endif

    // Render graph, Swap frames and Process Events
    pangolin::FinishFrame();  

    time0 = std::chrono::steady_clock::now();
  }

  return 0;
}
