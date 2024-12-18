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

#include <deque>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& p);

template <class T, class S, class C>
const S& getContainer(const std::priority_queue<T, S, C>& q) {
    // get the container within a priority queue
    struct HackedQueue : private std::priority_queue<T, S, C> {
        static const S& getContainer(const std::priority_queue<T, S, C>& qu) {
            return qu.*&HackedQueue::c;
        }
    };
    return HackedQueue::getContainer(q);
}

template <class T, class S, class C>
std::ostream& operator<<(std::ostream& out, const std::priority_queue<T, S, C>& q) {
    const auto& d = getContainer<T, S, C>(q);
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (it != std::prev(std::end(d)))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::pair<T1, T2>& p) {
    out << "(" << p.first << ", " << p.second << ")";
    return out;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::unordered_map<T1, T2>& d) {
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (std::next(it) != std::end(d))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, const std::map<T1, T2>& d) {
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (std::next(it) != std::end(d))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::deque<T>& d) {
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (std::next(it) != std::end(d))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::set<T>& d) {
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (std::next(it) != std::end(d))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::list<T>& d) {
    out << "<";
    for (auto it = std::begin(d); it != std::end(d); it++)
    {
        out << *it;
        if (std::next(it) != std::end(d))
            out << ", ";
    }
    out << ">";
    return out;
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v) {
    out << "[";
    for (size_t ii = 0; ii < v.size(); ii++)
    {
        out << v[ii];
        if (ii != (v.size() - 1))
            out << ", ";
    }
    out << "]";
    return out;
}

template <typename T>
auto vecSum(std::vector<T>& nums) {
    T sum = 0;
    for (const auto& e : nums)
        sum += e;
    return sum;
}

template <typename T>
bool isEqual(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size())
        return false;
    for (size_t ii = 0; ii < a.size(); ii++)
    {
        if (a[ii] != b[ii])
            return false;
    }
    return true;
}

template <typename IterStart, typename IterEnd>
bool isEqual(IterStart s1, IterEnd e1, IterStart s2, IterEnd e2) {
    if (std::distance(s1, e1) != std::distance(s2, e2))
        return false;
    for (; s1 != e1 && s2 != e2; s1++, s2++)
    {
        if (*s1 != *s2)
            return false;
    }
    return true;
}

// A hash function used to hash a pair of any kind
struct PairHasher {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
        const auto hash1 = std::hash<T1>{}(p.first);
        const auto hash2 = std::hash<T2>{}(p.second);

        if (hash1 != hash2)
        {
            return hash1 ^ hash2;
        }

        // If hash1 == hash2, their XOR is zero.
        return hash1;
    }
};