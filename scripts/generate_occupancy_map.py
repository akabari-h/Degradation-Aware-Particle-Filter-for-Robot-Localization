#!/usr/bin/env python3
"""
Generate occupancy grid map + distance transform from known SDF wall geometry.
Updated for 20m x 15m warehouse with obstacles.
Run once:  python3 generate_occupancy_map.py
Creates:   config/map.pgm, config/map.yaml, config/distance_map.npy
"""
import numpy as np
from PIL import Image
import os
from scipy.ndimage import distance_transform_edt

# =============================
#   World geometry (from SDF)
# =============================
# Room: 20m x 15m, origin at center
# x: -10.0 to +10.0
# y: -7.5 to +7.5

RESOLUTION = 0.05  # meters per pixel (5cm)
WALL_THICKNESS = 0.1  # meters

# World bounds (add padding outside walls)
X_MIN, X_MAX = -11.0, 11.0
Y_MIN, Y_MAX = -8.5, 8.5

# Map dimensions in pixels
WIDTH = int((X_MAX - X_MIN) / RESOLUTION)
HEIGHT = int((Y_MAX - Y_MIN) / RESOLUTION)

# Occupancy grid: 255 = free (white), 0 = occupied (black)
grid = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)


def world_to_pixel(x, y):
    px = int((x - X_MIN) / RESOLUTION)
    py = int((Y_MAX - y) / RESOLUTION)
    return px, py


def draw_wall(x_center, y_center, width_m, height_m):
    x1 = x_center - width_m / 2
    x2 = x_center + width_m / 2
    y1 = y_center - height_m / 2
    y2 = y_center + height_m / 2

    px1, py1 = world_to_pixel(x1, y2)
    px2, py2 = world_to_pixel(x2, y1)

    px1 = max(0, px1)
    py1 = max(0, py1)
    px2 = min(WIDTH - 1, px2)
    py2 = min(HEIGHT - 1, py2)

    grid[py1:py2 + 1, px1:px2 + 1] = 0


# =============================
#   Draw all walls from SDF
# =============================

# Outer walls
draw_wall(0, 7.55, 20.1, 0.1)       # North
draw_wall(0, -7.55, 20.1, 0.1)      # South
draw_wall(10.05, 0, 0.1, 15.1)      # East
draw_wall(-10.05, 0, 0.1, 15.1)     # West

# One shelf pair (upper center)
draw_wall(0.0, 5.5, 4.0, 0.6)       # Shelf A1
draw_wall(0.0, 4.2, 4.0, 0.6)       # Shelf A2

# Crates (center area)
draw_wall(-4.0, -2.0, 1.3, 1.3)     # Crate 1
draw_wall(4.0, -3.0, 1.1, 1.1)      # Crate 2

# Pillar (center)
draw_wall(0.0, -1.0, 0.4, 0.4)      # Pillar 1

# Mark outside-of-room as occupied
for py in range(HEIGHT):
    for px in range(WIDTH):
        wx = X_MIN + px * RESOLUTION
        wy = Y_MAX - py * RESOLUTION
        if wx < -10.0 or wx > 10.0 or wy < -7.5 or wy > 7.5:
            grid[py, px] = 0

# =============================
#   Save map files
# =============================
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config')
os.makedirs(output_dir, exist_ok=True)

pgm_path = os.path.join(output_dir, 'map.pgm')
img = Image.fromarray(grid)
img.save(pgm_path)
print(f'Saved occupancy grid: {pgm_path} ({WIDTH}x{HEIGHT} pixels)')

yaml_path = os.path.join(output_dir, 'map.yaml')
with open(yaml_path, 'w') as f:
    f.write(f"image: map.pgm\n")
    f.write(f"resolution: {RESOLUTION}\n")
    f.write(f"origin: [{X_MIN}, {Y_MIN}, 0.0]\n")
    f.write(f"negate: 0\n")
    f.write(f"occupied_thresh: 0.65\n")
    f.write(f"free_thresh: 0.196\n")
print(f'Saved map metadata: {yaml_path}')

# =============================
#   Precompute distance transform
# =============================
free_mask = grid > 200
dist_pixels = distance_transform_edt(free_mask)
dist_meters = dist_pixels * RESOLUTION

npy_path = os.path.join(output_dir, 'distance_map.npy')
np.save(npy_path, dist_meters)
print(f'Saved distance transform: {npy_path}')
print(f'  Shape: {dist_meters.shape}')
print(f'  Max distance: {dist_meters.max():.2f} m')
print(f'  Resolution: {RESOLUTION} m/pixel')

print('\nDone! Files saved in config/')