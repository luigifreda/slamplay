# ROS Compatibility Modules (No System ROS Required)

This folder contains ROS-oriented components that are integrated into the Slamplay build
without requiring a system-wide ROS installation.

The modules here are built against the in-repository `thirdparty/ros` dependency, which
provides the ROS interfaces needed by this project. 

The message-definition packages are not built; they are included here only as reference
`.msg` definitions for the ROS message types used in this repository.

## Purpose

- Keep ROS message and conversion utilities available in non-ROS environments.
- Reuse ROS-compatible data definitions (`*.msg`) and package metadata where useful.
- Allow selective build integration through the local CMake setup.

## Folder Contents

- `pointcloud_convert`
  - Point cloud conversion utilities used by this repository.
  - Plain CMake target integrated directly into the project build.

- `livox_ros_driver`
  - ROS-style package content (`msg`, `package.xml`, local `CMakeLists.txt`).
  - Kept in-tree for compatibility/reference and optional integration.

- `monitor_msgs`
  - ROS message definitions for monitor-related communication.
  - Organized as a standard ROS message package structure.

- `velodyne_msgs`
  - ROS message definitions for Velodyne-related data exchange.
  - Organized as a standard ROS message package structure.

## Build Behavior

The top-level `ros/CMakeLists.txt` controls which subfolders are built by default.

At the moment, only:

- `pointcloud_convert`

is enabled in the `FOLDERS` list and added with `add_subdirectory(...)`.

To include additional modules, add their folder names to `FOLDERS` in
`ros/CMakeLists.txt`.

## Notes

- You do not need to install ROS on the host machine to build the enabled targets here.
- The message packages included in this folder are already supported by `thirdparty/ros`,
  so the required message interfaces are available without external ROS message generation.
- ROS-like package files (`package.xml`, `msg/`) are preserved to keep compatibility with
  existing ROS workflows and tooling expectations.