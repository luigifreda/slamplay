# ROS Bridge Minimal

<!-- TOC -->

- [ROS Bridge Minimal](#ros-bridge-minimal)
  - [Why this exists](#why-this-exists)
  - [What this package provides](#what-this-package-provides)
  - [Supported message packages](#supported-message-packages)
  - [Adding new message types](#adding-new-message-types)
  - [Build](#build)
  - [Scope and limitations](#scope-and-limitations)

<!-- /TOC -->


This folder contains a minimal, self-contained ROS1-compatible C++ subset used to read ROS bag data in this project.

## Why this exists

- No full ROS installation is required.
- No `catkin` workspace is required.
- We only vendor the components needed for bag I/O and message serialization.

The package is intended for offline data extraction and integration in non-ROS build systems.

## What this package provides

- A standalone static library: `librosbag.a` (output in `thirdparty/ros/lib`).
- Core ROS C++ runtime pieces required by rosbag storage:
  - `cpp_common`
  - `rostime`
  - `roscpp_serialization`
  - `roscpp_traits`
  - `roslz4`
  - `console_bridge`
  - `rosbag_storage`
- Header-only message definitions under `include/` so existing ROS message types can be deserialized without a system ROS install.

## Supported message packages

The vendored headers include standard ROS message packages:

- `std_msgs`
- `sensor_msgs`
- `geometry_msgs`

and project/custom message packages:

- `livox_ros_driver` (`CustomMsg`, `CustomPoint`)
- `velodyne_msgs` (`VelodynePacket`, `VelodyneScan`, `VelodyneScanRaw`)
- `monitor_msgs` (`fault_info`, `fault_vec`)

In practice, bag topics using these message definitions are supported out of the box.

## Adding new message types

To decode a new topic type from bag files, add the generated ROS message headers to this vendored tree.

1. Generate ROS1 C++ headers for your `.msg` (typically using `gencpp` in any ROS-enabled environment).
2. Copy generated headers into `thirdparty/ros/include/<your_package>/`.
3. Copy all dependent message headers as well (for example `std_msgs/Header.h`, nested custom messages, etc.).
4. Ensure the generated header contains:
   - `ros::message_traits::MD5Sum`
   - `ros::message_traits::DataType`
   - `ros::message_traits::Definition`
   - `ros::serialization::Serializer`
5. Rebuild `thirdparty/ros` and your consumer target.

Important compatibility rule:

- The `MD5Sum` and `DataType` in the header must exactly match what is stored in the bag connection metadata. If they differ, deserialization will fail at runtime.

Quick verification:

- Open a bag containing the new topic and try instantiating that message type from `rosbag::MessageInstance`.
- If conversion succeeds and fields look valid, the message integration is complete.

## Build

From `thirdparty/ros`:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

After building, link against `lib/rosbag` and include headers from `thirdparty/ros/include` (plus the internal include paths exported by CMake).

## Scope and limitations

- This is not a complete ROS distribution.
- It focuses on bag storage, serialization, and message decoding needed by SlamPlay.
- ROS graph/runtime features (nodes, roscore, pub/sub transport, tooling) are intentionally not provided.