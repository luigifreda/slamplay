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

namespace Eigen {

#if __cplusplus < 201703L  // If C++ version is earlier than C++17

template <typename T>
using aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;

template <typename T>
using aligned_deque = std::deque<T, Eigen::aligned_allocator<T>>;

template <typename K, typename V>
using aligned_map = std::map<K, V, std::less<K>, Eigen::aligned_allocator<std::pair<K const, V>>>;

template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>,
                                                 Eigen::aligned_allocator<std::pair<K const, V>>>;

#else  // For C++17 and later, no need for aligned_allocator

template <typename T>
using aligned_vector = std::vector<T>;

template <typename T>
using aligned_deque = std::deque<T>;

template <typename K, typename V>
using aligned_map = std::map<K, V>;

template <typename K, typename V>
using aligned_unordered_map = std::unordered_map<K, V>;

#endif

}  // namespace Eigen
