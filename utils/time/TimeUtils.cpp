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
#include "time/TimeUtils.h"
#include "io/format.h"
#include "io/messages.h"

#include <fstream>
#include <iomanip>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#define UNUSED(x) (void)(x)

namespace slamplay {

// compute the median of an eigen vector
// Note: Changes the order of elements in the vector!
// Note: For even sized vectors we don't return the mean of the middle two, but
// simply the second one as is.
template <class Scalar, int Rows>
Scalar median_non_const(Eigen::Matrix<Scalar, Rows, 1>& vec) {
    static_assert(Rows != 0);
    if constexpr (Rows < 0) {
        MSG_ASSERT(vec.size() >= 1, "Cannot compute median of an empty vector");
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

ExecutionStats::Meta& ExecutionStats::add(const std::string& name,
                                          double value) {
    auto [it, new_item] = stats_.try_emplace(name);
    if (new_item) {
        order_.push_back(name);
        it->second.data_ = std::vector<double>();
    }
    std::get<std::vector<double>>(it->second.data_).push_back(value);
    return it->second;
}

ExecutionStats::Meta& ExecutionStats::add(const std::string& name,
                                          const Eigen::VectorXd& value) {
    auto [it, new_item] = stats_.try_emplace(name);
    if (new_item) {
        order_.push_back(name);
        it->second.data_ = std::vector<Eigen::VectorXd>();
    }
    std::get<std::vector<Eigen::VectorXd>>(it->second.data_).push_back(value);
    return it->second;
}

ExecutionStats::Meta& ExecutionStats::add(const std::string& name,
                                          const Eigen::VectorXf& value) {
    Eigen::VectorXd x = value.cast<double>();
    return add(name, x);
}

void ExecutionStats::merge_all(const ExecutionStats& other) {
    for (const auto& name : other.order_) {
        const auto& meta = other.stats_.at(name);
        std::visit(
            [&](auto& data) {
                for (auto v : data) {
                    add(name, v);
                }
            },
            meta.data_);
        stats_.at(name).set_meta(meta);
    }
}

namespace {  // helper
// ////////////////////////////////////////////////////////////////////////////
// overloads for generic lambdas
// See also: https://stackoverflow.com/q/55087826/1813258
template <class... Ts>
struct overload : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overload(Ts...) -> overload<Ts...>;
}  // namespace

void ExecutionStats::merge_sums(const ExecutionStats& other) {
    for (const auto& name : other.order_) {
        const auto& meta = other.stats_.at(name);
        std::visit(overload{[&](const std::vector<double>& data) {
                                Eigen::Map<const Eigen::VectorXd> map(data.data(),
                                                                      data.size());
                                add(name, map.sum());
                            },
                            [&](const std::vector<Eigen::VectorXd>& data) {
                                UNUSED(data);
                                // TODO: for now no-op
                            }},
                   meta.data_);
        stats_.at(name).set_meta(meta);
    }
}

void ExecutionStats::print() const {
    for (const auto& name : order_) {
        const auto& meta = stats_.at(name);

        std::visit(
            overload{
                [&](const std::vector<double>& data) {
                    Eigen::Map<const Eigen::VectorXd> map(data.data(), data.size());

                    // create a copy for median computation
                    Eigen::VectorXd vec = map;

                    if (meta.format_ == "ms") {
                        // convert seconds to milliseconds
                        vec *= 1000;
                    }

                    int count = vec.size();
                    // double sum = vec.sum();
                    double mean = vec.mean();
                    double stddev = std::sqrt(variance(vec));
                    double max = vec.maxCoeff();
                    double min = vec.minCoeff();

                    // double median = median_non_const(vec);

                    if (meta.format_ == "count") {
                        std::cout << "{:20} ({:>4}):{: 8.1f}+-{:.1f} [{}, {}]\n"_format(
                            name, count, mean, stddev, min, max);
                    } else if (meta.format_ != "none") {
                        std::cout
                            << "{:20} ({:>4}):{: 8.2f}+-{:.2f} [{:.2f}, {:.2f}]\n"_format(
                                   name, count, mean, stddev, min, max);
                    }
                },
                [&](const std::vector<Eigen::VectorXd>& data) {
                    int count = data.size();
                    std::cout << "{:20} ({:>4})\n"_format(name, count);
                }},
            meta.data_);
    }
}

bool ExecutionStats::save_json(const std::string& path) const {
    using json = nlohmann::json;
    json result;

    for (const auto& name : order_) {
        const auto& meta = stats_.at(name);

        std::visit(
            overload{[&](const std::vector<double>& data) { result[name] = data; },
                     [&](const std::vector<Eigen::VectorXd>& data) {
                         std::vector<int> indices;
                         std::vector<double> values;
                         for (const auto& v : data) {
                             indices.push_back(int(values.size()));
                             // values.insert(values.end(), v.begin(), v.end());
                             values.insert(values.end(), v.data(), v.data() + v.size());
                         }
                         std::string name_values = std::string(name) + "__values";
                         std::string name_indices = std::string(name) + "__index";
                         result[name_indices] = indices;
                         result[name_values] = values;
                     }},
            meta.data_);
    }

    constexpr bool save_as_json = false;
    constexpr bool save_as_ubjson = true;

    // save json
    if (save_as_json) {
        std::ofstream ofs(path);

        if (!ofs.is_open()) {
            std::cerr << "Could not save ExecutionStats to {}.\n"_format(path);
            return false;
        }

        ofs << std::setw(4) << result;  //!< pretty printing
        // ofs << result;  //!< no pretty printing

        std::cout << "Saved ExecutionStats to {}.\n"_format(path);
    }

    // save ubjson
    if (save_as_ubjson) {
        std::string ubjson_path =
            path.substr(0, path.find_last_of('.')) + ".ubjson";
        std::ofstream ofs(ubjson_path, std::ios_base::binary);

        if (!ofs.is_open()) {
            std::cerr << "Could not save ExecutionStats to {}.\n"_format(ubjson_path);
            return false;
        }

        json::to_ubjson(result, ofs);

        std::cout << "Saved ExecutionStats to {}.\n"_format(ubjson_path);
    }

    return true;
}

}  // namespace slamplay
