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
#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

namespace slamplay {

///	\namespace SignalUtils
///	\brief some signal utilities gathered in a class
///	\author
///	\note
namespace SignalUtils {

// set output to zero when fabs(input)< threshold
inline double deadZone(double in, const double threshold);

// set output to sign(input)*minValue when fabs(input)>=thresholdm fabs(input)<thresholdM
inline double minValueZone(double in, const double minValue, const double thresholdm, const double thresholdM);

template <typename T>
inline T sat(T val, T min, T max) {
    return std::min(std::max(val, min), max);
}

template <typename T>
inline int sign(const T a) {
    return ((a >= 0) ? 1 : -1);
}

};  // namespace SignalUtils

// set output to zero when fabs(input)< threshold
double SignalUtils::deadZone(double in, double threshold) {
    double out = in;
    if (fabs(in) < threshold)
    {
        out = 0;
    }
    return out;
}

// set output to sign(input)*value when fabs(input)>=thresholdm fabs(input)<thresholdM
double SignalUtils::minValueZone(double in, const double minValue, const double thresholdm, const double thresholdM) {
    double out = in;
    double fabsIn = fabs(in);

    if ((fabsIn < thresholdM) && (fabsIn >= thresholdm))
    {
        out = sign(in) * fabs(minValue);
    } else if (fabsIn < thresholdm)
        out = 0;

    return out;
}

}  // namespace slamplay