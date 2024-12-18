/*
 * This file is part of slamplay
 * Copyright (C) 2018-present Luigi Freda <luigifreda at gmail dot com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#pragma once

#include <fstream>
#include <iostream>
#include <list>
#include <vector>

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

namespace slamplay {

// compute the median of an eigen vector
// Note: Changes the order of elements in the vector!
// Note: For even sized vectors we don't return the mean of the middle two, but
// simply the second one as is.
template <class Scalar, int Rows>
Scalar median(const Eigen::Matrix<Scalar, Rows, 1>& vec) {
    static_assert(Rows != 0);
    if constexpr (Rows < 0) {
        BASALT_ASSERT(vec.size() >= 1);
    }
    int n = vec.size() / 2;
    std::nth_element(vec.begin(), vec.begin() + n, vec.end());
    return vec(n);
}

template <class Scalar, int N>
Scalar variance(const Eigen::Matrix<Scalar, N, 1>& vec) {
    static_assert(N != 0);
    const Eigen::Matrix<Scalar, N, 1> centered = vec.array() - vec.mean();
    return centered.squaredNorm() / Scalar(vec.size());
}

}  // namespace slamplay
