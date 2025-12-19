# Data Generation Config — Condensed Reference

This document provides a compact description of all configuration parameters.

---

## Top-Level

| Key                     | Type       | Description                                 |
| ----------------------- | ---------- | ------------------------------------------- |
| `scenario_name`         | string     | Human-readable scenario or experiment name  |
| `output_root`           | path       | Root directory for generated outputs        |
| `output_name`           | string     | Subdirectory / experiment identifier        |
| `copy_config_to_output` | bool       | Copy config JSON into output directory      |
| `output_size`           | [int, int] | Output image resolution `[width, height]`   |
| `num_frames`            | int        | Number of frames to generate                |
| `num_subframes`         | int        | Subframes per frame (temporal accumulation) |

---

## Camera

| Key              | Type                  | Description                  |
| ---------------- | --------------------- | ---------------------------- |
| `position`       | [float, float, float] | Camera world position (xyz)        |
| `look_at`        | [float, float, float] | Target point camera looks at (xyz) |
| `focal_length`   | float                 | Focal length (mm)            |
| `focus_distance` | float                 | Focus distance               |

---

## Light

| Key               | Type           | Description                           |
| ----------------- | -------------- | ------------------------------------- |
| `num_lights`      | int            | Number of lights                      |
| `type`            | string         | Light type (`distant`, `point`, etc.) |
| `xy_range`        | [float, float] | X/Y position range                    |
| `z_range`         | [float, float] | Z position range                      |
| `intensity`       | [float, float] | Intensity range                       |
| `exposure`        | [float, float] | Exposure multiplier range             |
| `randomize_color` | bool           | Enable random light colors            |
| `r_range`         | [float, float] | Red channel range                     |
| `g_range`         | [float, float] | Green channel range                   |
| `b_range`         | [float, float] | Blue channel range                    |

---

## Background

| Key                   | Type           | Description                  |
| --------------------- | -------------- | ---------------------------- |
| `depth`               | float          | Background plane distance    |
| `plane_angle_range`   | [float, float] | Plane XY rotation range (deg)   |
| `scale`               | float          | Plane scale                  |
| `show_plane`          | bool           | Render background plane      |
| `spawn_objects`       | bool           | Spawn background objects     |
| `object_plane_offset` | float          | Object plane offset from background plane     |
| `check_collisions`    | int            | Collision checking mode [DOCS](https://docs.omniverse.nvidia.com/py/replicator/1.12.27/source/extensions/omni.replicator.core/docs/API.html#omni.replicator.core.randomizer.scatter_2d)      |
| `object_count`        | int            | Number of background objects |
| `object_scale`        | [float, float] | Object uniform scale range   |
| `dim_scale_bounds`    | [float, float] | Per-axis scale bounds        |
| `objects`             | bool         | Primitive types to include as background objects      |

### Background Materials

| Key               | Type           | Description                   |
| ----------------- | -------------- | ----------------------------- |
| `root`            | path           | Material root directory       |
| `include_folders` | list[string]   | Material subfolders to sample |
| `variants`        | int            | Variants per object           |
| `roughness_range` | [float, float] | Roughness range [0-1]               |
| `metallic_range`  | [float, float] | Metallic range [0-1]                |
| `specular_range`  | [float, float] | Specular range [0-1]                |

---

## Foreground

| Key                    | Type                  | Description                 |
| ---------------------- | --------------------- | --------------------------- |
| `spawn_objects`        | bool                  | Spawn foreground objects    |
| `model_root`           | string (path)         | Root directory for models   |
| `visible_object_count` | int                   | Average visible objects per frame   |
| `instance_per_object`  | int                   | Number of copies to create for each object        |
| `objects`              | map (model path: class label)| Model path is wrt model_root |
| `random_position`      | bool                  | Randomize position          |
| `random_rotation`      | bool                  | Randomize rotation          |
| `random_color`         | bool                  | Randomize solid color textures        |
| `random_materials`     | bool                  | Randomize materials         |
| `spawn_vfov`           | float                 | VFOV for spawn volume       |
| `z_range`              | [float, float]        | Z (depth) placement bounds          |
| `default_pos`          | [float, float, float] | Default position            |
| `default_rot`          | [float, float, float] | Default rotation (deg)      |
| `default_scale`        | float                 | Default scale               |

### Foreground Materials

*(Only applies if `random_materials` is true, same fields and meaning as Background Materials)*

---

## Output Types

Each flag enables generation of the corresponding output per frame.

| Key                       | Description                  |
| ------------------------- | ---------------------------- |
| `rgb`                     | RGB image                    |
| `camera_params`           | Camera intrinsics/extrinsics |
| `bounding_box_2d_tight`   | Tight 2D bounding boxes      |
| `bounding_box_2d_loose`   | Loose 2D bounding boxes      |
| `semantic_segmentation`   | Semantic labels/masks              |
| `instance_segmentation`   | Instance labels/masks              |
| `distance_to_camera`      | Depth (camera space)         |
| `distance_to_image_plane` | Depth (image plane)          |
| `bounding_box_3d`         | 3D bounding boxes            |
| `occlusion`               | Occlusion metrics            |
| `normals`                 | Surface normals              |
| `pointclouds`             | Point cloud output           |

---

## Notes
* Units must match renderer conventions, by default Replicator uses 100 units_per_meter (centimeters).
* Data generation runs are automatically given a numeric suffix, so you can safely generate multiple batches of data using the same config file / output_name and they will be output to separate output folders (i.e. 'test_001', 'test_002', etc.).
