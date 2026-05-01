#!/usr/bin/env python3
"""
Particle Cloud Visualizer
==========================
Subscribes to particle cloud, estimated pose, and ground truth.
Saves snapshots of particles on the occupancy map at regular intervals.

Usage:
  ros2 run rsn_p plot_particles.py

Saves images to ~/rsn_project/results/particles/
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Float64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import os
import yaml
from ament_index_python.packages import get_package_share_directory


class ParticlePlotter(Node):

    def __init__(self):
        super().__init__('particle_plotter')

        # Load map
        pkg_share = get_package_share_directory('rsn_p')
        map_yaml_path = os.path.join(pkg_share, 'config', 'map.yaml')
        with open(map_yaml_path, 'r') as f:
            map_meta = yaml.safe_load(f)

        self.resolution = map_meta['resolution']
        self.origin = map_meta['origin']

        map_pgm_path = os.path.join(pkg_share, 'config', 'map.pgm')
        self.map_img = np.array(PILImage.open(map_pgm_path))
        self.map_height, self.map_width = self.map_img.shape

        # Load marker positions
        marker_yaml_path = os.path.join(pkg_share, 'config', 'marker_positions.yaml')
        with open(marker_yaml_path, 'r') as f:
            marker_data = yaml.safe_load(f)
        self.markers = [(m['x'], m['y']) for m in marker_data['markers']]

        # Output directory
        self.output_dir = os.path.expanduser('~/rsn_project/results/particles')
        os.makedirs(self.output_dir, exist_ok=True)

        # State
        self.particles = None
        self.est_pose = None
        self.gt_pose = None
        self.spread = None
        self.alpha_cam = None
        self.alpha_ult = None
        self.snapshot_count = 0

        # Trajectory trails
        self.est_trail = []
        self.gt_trail = []

        # Subscribers
        self.create_subscription(
            PoseArray, '/da_pf/particles', self.particles_cb, 10)
        self.create_subscription(
            PoseStamped, '/da_pf/pose', self.est_cb, 10)
        self.create_subscription(
            PoseStamped, '/model/rover/pose', self.gt_cb, 10)
        self.create_subscription(
            Float64, '/da_pf/particle_spread', self.spread_cb, 10)
        self.create_subscription(
            Float64, '/confidence/camera', self.cam_conf_cb, 10)
        self.create_subscription(
            Float64, '/confidence/ultrasonic', self.ult_conf_cb, 10)

        # Save snapshot every 5 seconds
        self.timer = self.create_timer(5.0, self.save_snapshot)

        self.get_logger().info(f'Particle plotter started. Saving to {self.output_dir}')

    def particles_cb(self, msg):
        self.particles = [(p.position.x, p.position.y) for p in msg.poses]

    def est_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.est_pose = (x, y)
        self.est_trail.append((x, y))

    def gt_cb(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.gt_pose = (x, y)
        self.gt_trail.append((x, y))

    def spread_cb(self, msg):
        self.spread = msg.data

    def cam_conf_cb(self, msg):
        self.alpha_cam = msg.data

    def ult_conf_cb(self, msg):
        self.alpha_ult = msg.data

    def save_snapshot(self):
        if self.particles is None or self.est_pose is None:
            self.get_logger().info('Waiting for particle data...')
            return

        self.snapshot_count += 1

        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        # Draw occupancy map
        ax.imshow(self.map_img, cmap='gray', origin='upper',
                  extent=[self.origin[0],
                          self.origin[0] + self.map_width * self.resolution,
                          self.origin[1],
                          self.origin[1] + self.map_height * self.resolution])

        # Draw ArUco markers
        for mx, my in self.markers:
            ax.plot(mx, my, 's', color='purple', markersize=8, zorder=5)

        # Draw particle cloud
        px = [p[0] for p in self.particles]
        py = [p[1] for p in self.particles]
        ax.scatter(px, py, c='cyan', s=3, alpha=0.5, zorder=3, label='Particles')

        # Draw estimated pose trail
        if len(self.est_trail) > 1:
            ex = [p[0] for p in self.est_trail]
            ey = [p[1] for p in self.est_trail]
            ax.plot(ex, ey, 'b-', linewidth=1.5, alpha=0.7, label='DA-PF estimate')

        # Draw ground truth trail
        if len(self.gt_trail) > 1:
            gx = [p[0] for p in self.gt_trail]
            gy = [p[1] for p in self.gt_trail]
            ax.plot(gx, gy, 'g-', linewidth=1.5, alpha=0.7, label='Ground truth')

        # Current positions (large dots)
        ax.plot(self.est_pose[0], self.est_pose[1], 'bo',
                markersize=10, zorder=6)
        if self.gt_pose:
            ax.plot(self.gt_pose[0], self.gt_pose[1], 'go',
                    markersize=10, zorder=6)

        # Info text
        info = f'Snapshot {self.snapshot_count} | t={self.snapshot_count * 5}s'
        if self.spread is not None:
            info += f' | Spread: {self.spread:.2f}m'
        if self.alpha_cam is not None:
            info += f' | a_cam: {self.alpha_cam:.2f}'
        if self.alpha_ult is not None:
            info += f' | a_ult: {self.alpha_ult:.2f}'

        ax.set_title(info, fontsize=13)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xlim(-11, 11)
        ax.set_ylim(-8.5, 8.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

        filepath = os.path.join(
            self.output_dir, f'snapshot_{self.snapshot_count:03d}.png')
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close(fig)

        self.get_logger().info(
            f'Saved {filepath} | Particles: {len(self.particles)} | '
            f'Spread: {self.spread:.2f}m' if self.spread else f'Saved {filepath}')


def main(args=None):
    rclpy.init(args=args)
    node = ParticlePlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()