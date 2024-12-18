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

// A very simple custom container type.
template <typename T>
struct MyContainer {
    T* data;
    size_t size;

    MyContainer(size_t size_) : data(new T[size_]), size(size_) {}

    // For demonstration purposes: This container can't be copied.
    MyContainer(const MyContainer&) = delete;

    ~MyContainer() {
        delete[] data;
    }
};

// A custom vector type.
struct MyVec3 {
    float x, y, z;
};

/// Adapts `MyContainer<MyVec3>` to a `Collection<Position3D>`.
///
/// With this in place, `Collection<Position3D>` can be constructed from a `MyContainer<MyVec3>`!
template <>
struct rerun::CollectionAdapter<rerun::Position3D, MyContainer<MyVec3>> {
    // Creating a Collection from a non-temporary is done by casting & borrowing binary compatible data.
    Collection<rerun::Position3D> operator()(const MyContainer<MyVec3>& container) {
        return Collection<rerun::Position3D>::borrow(container.data, container.size);
    }

    // For temporaries we have to do a copy since the pointer doesn't live long enough.
    // If you don't implement this, the other overload may be used for temporaries and cause
    // undefined behavior.
    Collection<rerun::Position3D> operator()(MyContainer<MyVec3>&& container) {
        std::vector<rerun::Position3D> components(container.size);
        for (size_t i = 0; i < container.size; ++i) {
            components[i] =
                rerun::Position3D(container.data[i].x, container.data[i].y, container.data[i].z);
        }
        return Collection<rerun::Position3D>::take_ownership(std::move(components));
    }
};

int main() {
    // Create a new `RecordingStream` which sends data over TCP to the viewer process.
    const auto rec = rerun::RecordingStream("rerun_example_custom_component_adapter");
    rec.spawn().exit_on_failure();

    // Construct some data in a custom format.
    MyContainer<MyVec3> points(3);
    points.data[0] = MyVec3{0.0f, 0.0f, 0.0f};
    points.data[1] = MyVec3{1.0f, 0.0f, 0.0f};
    points.data[2] = MyVec3{0.0f, 1.0f, 0.0f};

    // Log the "my_points" entity with our data, using the `Points3D` archetype.
    // Of course you can mix and match built-in types and custom types on the same archetype.
    rec.log("my_points", rerun::Points3D(points).with_labels({"a", "b", "c"}));

    return 0;
}
