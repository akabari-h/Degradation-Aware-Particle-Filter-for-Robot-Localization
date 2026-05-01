#!/usr/bin/env python3
"""
Generate 8 ArUco marker PNGs and their Gazebo model directories.
Run once:  python3 generate_aruco_markers.py
Creates:   models/aruco_marker_0/ ... models/aruco_marker_7/
"""
import cv2
import numpy as np
import os

# ArUco dictionary — 6x6 with 250 possible IDs
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Where to create model folders (run from rsn_p package root)
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
MARKER_SIZE_PX = 500  # pixels (larger = sharper at distance)
NUM_MARKERS = 16

for marker_id in range(NUM_MARKERS):
    # Generate marker image (works with both old and new OpenCV)
    try:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, MARKER_SIZE_PX)
    except AttributeError:
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, MARKER_SIZE_PX)

    # Add white border (helps detection)
    bordered = cv2.copyMakeBorder(marker_img, 30, 30, 30, 30,
                                   cv2.BORDER_CONSTANT, value=255)

    # Create model directory structure
    model_name = f'aruco_marker_{marker_id}'
    model_dir = os.path.join(MODELS_DIR, model_name)
    texture_dir = os.path.join(model_dir, 'materials', 'textures')
    os.makedirs(texture_dir, exist_ok=True)

    # Save PNG
    png_path = os.path.join(texture_dir, f'aruco_{marker_id}.png')
    cv2.imwrite(png_path, bordered)
    print(f'Saved {png_path}')

    # Write model.config
    config_path = os.path.join(model_dir, 'model.config')
    with open(config_path, 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <description>ArUco marker ID {marker_id}</description>
</model>
""")

    # Write model.sdf (using simple diffuse material — works with ogre and ogre2)
    sdf_path = os.path.join(model_dir, 'model.sdf')
    with open(sdf_path, 'w') as f:
        f.write(f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box><size>0.5 0.005 0.5</size></box>
        </geometry>
        <material>
          <ambient>1 1 1 1</ambient>
          <diffuse>1 1 1 1</diffuse>
          <specular>0 0 0 1</specular>
          <pbr>
            <metal>
              <albedo_map>model://{model_name}/materials/textures/aruco_{marker_id}.png</albedo_map>
              <metalness>0.0</metalness>
              <roughness>1.0</roughness>
            </metal>
          </pbr>
        </material>
      </visual>
      <collision name="collision">
        <geometry>
          <box><size>0.5 0.005 0.5</size></box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
""")

print(f'\nDone! Generated {NUM_MARKERS} marker models in {MODELS_DIR}')