#!/usr/bin/env python3
"""
Degradation-Aware Particle Filter (DA-PF) Node
================================================
Fuses camera (ArUco landmarks), ultrasonic (range), and IMU/odometry
with confidence-based weighting for robust localization under
sensor degradation.

Update step:
  w_i = L_camera^(α_cam) × L_ultrasonic^(α_ult)

When α=1: sensor contributes fully (healthy)
When α=0: L^0=1, sensor contributes nothing (degraded)
Fallback: IMU dead reckoning when both sensors fail
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
import os
from ament_index_python.packages import get_package_share_directory
import math
from tf_transformations import quaternion_from_euler


class ParticleFilterNode(Node):

    def __init__(self):
        super().__init__('particle_filter')

        # ===========================
        #   Parameters
        # ===========================
        self.declare_parameter('num_particles', 500)

        # Odometry motion model noise
        self.declare_parameter('alpha1', 0.1)
        self.declare_parameter('alpha2', 0.05)
        self.declare_parameter('alpha3', 0.1)
        self.declare_parameter('alpha4', 0.05)

        # Sensor model std devs
        self.declare_parameter('sigma_range', 1.0)
        self.declare_parameter('sigma_bearing', 0.5)
        self.declare_parameter('sigma_ultrasonic', 0.5)

        # Resampling threshold
        self.declare_parameter('resample_threshold', 0.5)

        # ===========================
        #   Load map and markers
        # ===========================
        pkg_share = get_package_share_directory('rsn_p')

        map_yaml_path = os.path.join(pkg_share, 'config', 'map.yaml')
        with open(map_yaml_path, 'r') as f:
            map_meta = yaml.safe_load(f)

        self.map_resolution = map_meta['resolution']
        self.map_origin = map_meta['origin']

        dist_map_path = os.path.join(pkg_share, 'config', 'distance_map.npy')
        self.distance_map = np.load(dist_map_path)
        self.get_logger().info(
            f'Loaded distance map: {self.distance_map.shape}, '
            f'resolution: {self.map_resolution} m/px'
        )

        from PIL import Image as PILImage
        map_pgm_path = os.path.join(pkg_share, 'config', 'map.pgm')
        map_img = np.array(PILImage.open(map_pgm_path))
        self.free_mask = map_img > 200
        self.map_height, self.map_width = self.free_mask.shape

        marker_yaml_path = os.path.join(pkg_share, 'config', 'marker_positions.yaml')
        with open(marker_yaml_path, 'r') as f:
            marker_data = yaml.safe_load(f)

        self.marker_map = {}
        for m in marker_data['markers']:
            self.marker_map[m['id']] = (m['x'], m['y'])
        self.get_logger().info(f'Loaded {len(self.marker_map)} marker positions')

        # ===========================
        #   Initialize particles
        # ===========================
        N = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.num_particles = N
        self.particles = np.zeros((N, 3))
        self.weights = np.ones(N) / N
        self._initialize_particles()

        # ===========================
        #   ArUco detector
        # ===========================
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params
            )
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        # Camera intrinsics (800x600, hfov=1.39626 rad = 80 deg)
        self.img_width = 800
        self.img_height = 600
        self.hfov = 1.39626
        self.fx = self.img_width / (2.0 * math.tan(self.hfov / 2.0))

        # ===========================
        #   State variables
        # ===========================
        self.prev_odom = None
        self.alpha_cam = 1.0
        self.alpha_ult = 1.0
        self.latest_markers = []
        self.latest_ult_range = None
        self.initialized = False

        # ===========================
        #   Subscribers
        # ===========================
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.cam_sub = self.create_subscription(
            Image, '/camera/degraded', self.camera_callback, 10)
        self.ult_sub = self.create_subscription(
            LaserScan, '/ultrasonic/degraded', self.ultrasonic_callback, 10)
        self.cam_conf_sub = self.create_subscription(
            Float64, '/confidence/camera', self.cam_conf_callback, 10)
        self.ult_conf_sub = self.create_subscription(
            Float64, '/confidence/ultrasonic', self.ult_conf_callback, 10)

        # ===========================
        #   Publishers
        # ===========================
        self.pose_pub = self.create_publisher(PoseStamped, '/da_pf/pose', 10)
        self.particles_pub = self.create_publisher(PoseArray, '/da_pf/particles', 10)
        self.spread_pub = self.create_publisher(Float64, '/da_pf/particle_spread', 10)

        # Main loop at 10 Hz
        self.timer = self.create_timer(0.1, self.filter_loop)

        self.get_logger().info(f'DA-PF started with {N} particles (Camera + Ultrasonic + IMU)')

    # ===========================
    #   Initialization
    # ===========================
    def _initialize_particles(self):
        """Initialize particles near the known start position (0, 0)."""
        N = self.num_particles
        self.particles[:, 0] = np.random.normal(0.0, 0.5, size=N)
        self.particles[:, 1] = np.random.normal(0.0, 0.5, size=N)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, size=N)
        self.weights = np.ones(N) / N
        self.get_logger().info('Particles initialized near start position (0, 0)')

    # ===========================
    #   Callbacks
    # ===========================
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny, cosy)

        if self.prev_odom is None:
            self.prev_odom = (x, y, theta)
            self.initialized = True
            return
        self.prev_odom = (x, y, theta)

    def camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        if self.use_new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

        detected = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.marker_map:
                    cx = np.mean(corners[i][0][:, 0])
                    bearing = math.atan2(
                        self.img_width / 2.0 - cx, self.fx)
                    marker_px_width = (np.max(corners[i][0][:, 0])
                                       - np.min(corners[i][0][:, 0]))
                    if marker_px_width > 5:
                        obs_range = (0.5 * self.fx) / marker_px_width
                    else:
                        obs_range = 10.0
                    detected.append((int(marker_id), bearing, obs_range))
        self.latest_markers = detected

    def ultrasonic_callback(self, msg):
        if len(msg.ranges) > 0:
            r = msg.ranges[0]
            if msg.range_min <= r <= msg.range_max:
                self.latest_ult_range = r
            else:
                self.latest_ult_range = None

    def cam_conf_callback(self, msg):
        self.alpha_cam = msg.data

    def ult_conf_callback(self, msg):
        self.alpha_ult = msg.data

    # ===========================
    #   Particle Filter Steps
    # ===========================
    def predict(self, delta_x, delta_y, delta_theta):
        """Predict step: propagate particles using odometry motion model."""
        a1 = self.get_parameter('alpha1').get_parameter_value().double_value
        a2 = self.get_parameter('alpha2').get_parameter_value().double_value
        a3 = self.get_parameter('alpha3').get_parameter_value().double_value
        a4 = self.get_parameter('alpha4').get_parameter_value().double_value
        N = self.num_particles

        trans = math.sqrt(delta_x ** 2 + delta_y ** 2)
        if trans < 1e-6:
            rot1 = 0.0
            rot2 = delta_theta
        else:
            rot1 = math.atan2(delta_y, delta_x) - self.particles[:, 2]
            rot1 = np.arctan2(np.sin(rot1), np.cos(rot1))

        if isinstance(rot1, float):
            rot1_arr = np.full(N, rot1)
        else:
            rot1_arr = rot1

        rot2 = delta_theta - rot1_arr

        rot1_noisy = rot1_arr + np.random.normal(
            0, a1 * np.abs(rot1_arr) + a2 * abs(trans), N)
        trans_noisy = trans + np.random.normal(
            0, a3 * abs(trans) + a4 * (np.abs(rot1_arr) + np.abs(rot2)), N)
        rot2_noisy = rot2 + np.random.normal(
            0, a1 * np.abs(rot2) + a2 * abs(trans), N)

        self.particles[:, 0] += trans_noisy * np.cos(
            self.particles[:, 2] + rot1_noisy)
        self.particles[:, 1] += trans_noisy * np.sin(
            self.particles[:, 2] + rot1_noisy)
        self.particles[:, 2] += rot1_noisy + rot2_noisy
        self.particles[:, 2] = np.arctan2(
            np.sin(self.particles[:, 2]),
            np.cos(self.particles[:, 2]))

    def update(self):
        """
        Update step: score particles using camera and ultrasonic.
        w_i = L_camera^(α_cam) × L_ultrasonic^(α_ult)
        """
        sigma_r = self.get_parameter('sigma_range').get_parameter_value().double_value
        sigma_b = self.get_parameter('sigma_bearing').get_parameter_value().double_value
        sigma_u = self.get_parameter('sigma_ultrasonic').get_parameter_value().double_value
        N = self.num_particles

        # ---- Camera likelihood ----
        L_cam = np.ones(N)

        for marker_id, obs_bearing, obs_range in self.latest_markers:
            if marker_id not in self.marker_map:
                continue
            mx, my = self.marker_map[marker_id]
            dx = mx - self.particles[:, 0]
            dy = my - self.particles[:, 1]
            exp_range = np.sqrt(dx ** 2 + dy ** 2)
            exp_bearing = np.arctan2(dy, dx) - self.particles[:, 2]
            bearing_diff = np.arctan2(
                np.sin(obs_bearing - exp_bearing),
                np.cos(obs_bearing - exp_bearing))
            range_diff = obs_range - exp_range
            L_cam *= np.exp(-0.5 * (range_diff / sigma_r) ** 2)
            L_cam *= np.exp(-0.5 * (bearing_diff / sigma_b) ** 2)

        # ---- Ultrasonic likelihood ----
        L_ult = np.ones(N)

        if self.latest_ult_range is not None:
            px = ((self.particles[:, 0] - self.map_origin[0])
                  / self.map_resolution).astype(int)
            py = (self.map_height
                  - (self.particles[:, 1] - self.map_origin[1])
                  / self.map_resolution).astype(int)
            px = np.clip(px, 0, self.map_width - 1)
            py = np.clip(py, 0, self.map_height - 1)
            exp_range = self.distance_map[py, px]
            range_diff = self.latest_ult_range - exp_range
            L_ult = (0.95 * np.exp(-0.5 * (range_diff / sigma_u) ** 2)
                     + 0.05 * (1.0 / 4.0))

        # ---- Confidence-weighted fusion ----
        # w_i = L_camera^α_cam × L_ultrasonic^α_ult
        self.weights *= np.power(L_cam, self.alpha_cam)
        self.weights *= np.power(L_ult, self.alpha_ult)
        self.weights += 1e-300
        self.weights /= np.sum(self.weights)

    def resample(self):
        """Low-variance resampling when effective sample size is low."""
        N = self.num_particles
        threshold_frac = (
            self.get_parameter('resample_threshold')
            .get_parameter_value().double_value)
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff > threshold_frac * N:
            return

        positions = (np.arange(N) + np.random.random()) / N
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        indices = np.clip(indices, 0, N - 1)
        self.particles = self.particles[indices].copy()
        self.weights = np.ones(N) / N

    # ===========================
    #   Main filter loop
    # ===========================
    def filter_loop(self):
        """Run predict -> update -> resample at 10 Hz."""
        if not self.initialized or self.prev_odom is None:
            return

        curr_x, curr_y, curr_theta = self.prev_odom

        if not hasattr(self, '_last_x'):
            self._last_x = curr_x
            self._last_y = curr_y
            self._last_theta = curr_theta
            return

        delta_x = curr_x - self._last_x
        delta_y = curr_y - self._last_y
        delta_theta = curr_theta - self._last_theta
        delta_theta = math.atan2(
            math.sin(delta_theta), math.cos(delta_theta))

        self._last_x = curr_x
        self._last_y = curr_y
        self._last_theta = curr_theta

        if abs(delta_x) > 1e-5 or abs(delta_y) > 1e-5 or abs(delta_theta) > 1e-4:
            self.predict(delta_x, delta_y, delta_theta)

        self.update()
        self.resample()
        self._publish_estimated_pose()
        self._publish_particles()
        self._publish_spread()

    # ===========================
    #   Publishing
    # ===========================
    def _publish_estimated_pose(self):
        est_x = np.average(self.particles[:, 0], weights=self.weights)
        est_y = np.average(self.particles[:, 1], weights=self.weights)
        est_theta = math.atan2(
            np.average(np.sin(self.particles[:, 2]), weights=self.weights),
            np.average(np.cos(self.particles[:, 2]), weights=self.weights))

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = est_x
        msg.pose.position.y = est_y
        msg.pose.position.z = 0.0
        q = quaternion_from_euler(0, 0, est_theta)
        msg.pose.orientation.x = q[0]
        msg.pose.orientation.y = q[1]
        msg.pose.orientation.z = q[2]
        msg.pose.orientation.w = q[3]
        self.pose_pub.publish(msg)

    def _publish_particles(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        step = max(1, self.num_particles // 100)
        for i in range(0, self.num_particles, step):
            pose = Pose()
            pose.position.x = self.particles[i, 0]
            pose.position.y = self.particles[i, 1]
            pose.position.z = 0.0
            q = quaternion_from_euler(0, 0, self.particles[i, 2])
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]
            msg.poses.append(pose)
        self.particles_pub.publish(msg)

    def _publish_spread(self):
        var_x = np.var(self.particles[:, 0])
        var_y = np.var(self.particles[:, 1])
        spread = math.sqrt(var_x + var_y)
        msg = Float64()
        msg.data = spread
        self.spread_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()