# LIO SLAM applications

Offline LiDAR-inertial mapping and localization binaries. Implementations live in `full_slam/lio_slam/`; sensor and dataset settings are in `config/lio_slam/`.

Build from the repo root, then run from the build tree:

```bash
cd build/full_slam/apps/lio_slam
```

## Configuration

Edit `config/lio_slam/mapping.yaml` before running:

- `bag_path` — input rosbag
- `lio_yaml` — sensor preset (`velodyne.yaml`, `avia.yaml`, `velodyne_nclt.yaml`, …)
- `origin` — map origin (UTM); required for localization
- `map_data` — tiled map directory (for `run_localization_fusion_offline`)

All pipeline apps accept `--config_yaml` (default: `config/lio_slam/mapping.yaml`).

## Pipeline overview

| Step | Executable | Output |
|------|------------|--------|
| 1. Frontend | `run_lio_step_frontend` | `results/ad/lio_slam/map/` (keyframes, scans) |
| 2. Optimization (stage 1) | `run_lio_step_optimization --stage=1` | updated `keyframes.txt`, `opti_pose_1_` |
| 3. Loop closure | `run_lio_step_loop_closure` | loop constraints |
| 4. Optimization (stage 2) | `run_lio_step_optimization --stage=2` | `opti_pose_2_` |
| — | `run_lio_mapping` | runs steps 1–4 in sequence |

Run steps in order when using the step binaries; each stage reads/writes under `results/ad/lio_slam/map/`.

## Executables

### `run_lio_mapping`

Full offline SLAM: frontend → opti stage 1 → loop closure → opti stage 2.

```bash
./run_lio_mapping
./run_lio_mapping --config_yaml=/path/to/mapping.yaml
```

### `run_lio_step_frontend`

LIO front-end only: keyframe extraction and lidar poses into `results/ad/lio_slam/map/`.

```bash
./run_lio_step_frontend
```

### `run_lio_step_optimization`

Pose-graph optimization. Stage 1 runs before loop closure; stage 2 uses loop constraints.

```bash
./run_lio_step_optimization --stage=1
./run_lio_step_optimization --stage=2
```

### `run_lio_step_loop_closure`

Detect loop closures and add constraints. Requires frontend and optimization stage 1.

```bash
./run_lio_step_loop_closure
```

### `dump_lio_map`

Merge keyframe scans into a single `map.pcd` (voxel-downsampled).

```bash
./dump_lio_map
./dump_lio_map --pose_source=opti2 --voxel_size=0.1
```

`--pose_source`: `lidar` (default), `rtk`, `opti1`, `opti2`.  
`--dump_to`: output directory (default: `results/ad/lio_slam/map/`).

### `split_lio_map`

Split the optimized map into grid tiles under `results/ad/lio_slam/map_data/` (one PCD per cell plus `map_index.txt`). Uses `opti_pose_2_`. Run after mapping before localization.

```bash
./split_lio_map
./split_lio_map --voxel_size=0.1 --grid_size=100.0 --map_path=results/ad/lio_slam/map
```

`--grid_size`: tile width in meters (default: `100.0`).  
Set `map_data` in `mapping.yaml` to this directory for fusion.

### `dump_lio_map_data`

Load split map tiles, assign a distinct color per grid cell, merge into `map_colored.pcd`, and open a PCL viewer. Run after `split_lio_map`.

```bash
./dump_lio_map_data
./dump_lio_map_data --map_data_path=results/ad/lio_slam/map_data
```

`--map_data_path`: tiled map directory (`map_index.txt` + `{gx}_{gy}.pcd` files; default: `results/ad/lio_slam/map_data/`).  
`--dump_to`: output directory for `map_colored.pcd` (default: same as `--map_data_path`).

### `run_localization_fusion_offline`

Offline RTK + lidar + IMU localization against a pre-built tiled map. Expects `map_data` and `origin` in the config; run `split_lio_map` first.

```bash
./run_localization_fusion_offline
```

## Typical workflows

**Full map build**

```bash
./run_lio_mapping
./dump_lio_map --pose_source=opti2
./split_lio_map
./dump_lio_map_data   # optional: visualize tiles by grid cell
```

**Debug one stage**

```bash
./run_lio_step_frontend
./run_lio_step_optimization --stage=1
./run_lio_step_loop_closure
./run_lio_step_optimization --stage=2
```

**Localize on existing map**

1. Complete mapping and `split_lio_map`
2. Point `map_data` and `origin` in `mapping.yaml` at your map
3. `./run_localization_fusion_offline`
