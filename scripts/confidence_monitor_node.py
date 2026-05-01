#!/usr/bin/env python3
"""
Confidence Monitor Node
========================


Camera confidence:
  α_cam = markers_detected / max_markers_in_clean_conditions
  Based on ArUco marker detection count from the degraded camera image.

Ultrasonic confidence:
  α_ult = exp(-k * variance_of_recent_readings)
  Based on variance of range readings over a 2-second sliding window.


"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque


class ConfidenceMonitorNode(Node):

    def __init__(self):
        super().__init__('confidence_monitor')

        # Parameters
        self.declare_parameter('max_markers_clean', 3)
        self.declare_parameter('variance_k', 50.0)
        self.declare_parameter('window_duration', 2.0)

        # ArUco detector
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        try:
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.aruco_params)
            self.use_new_api = True
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

        # Ultrasonic sliding window
        self.range_buffer = deque()

        # State
        self.alpha_cam = 1.0
        self.alpha_ult = 1.0
        self.latest_markers_detected = 0

        # Subscribers
        self.cam_sub = self.create_subscription(
            Image, '/camera/degraded', self.camera_callback, 10)
        self.ult_sub = self.create_subscription(
            LaserScan, '/ultrasonic/degraded', self.ultrasonic_callback, 10)

        # Publishers
        self.cam_conf_pub = self.create_publisher(
            Float64, '/confidence/camera', 10)
        self.ult_conf_pub = self.create_publisher(
            Float64, '/confidence/ultrasonic', 10)

        # 10 Hz publish timer
        self.timer = self.create_timer(0.1, self.publish_confidence)

        self.get_logger().info('Confidence monitor started (Camera + Ultrasonic)')

    def camera_callback(self, msg):
        """Detect ArUco markers and compute camera confidence."""
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

        if ids is not None:
            self.latest_markers_detected = len(ids)
        else:
            self.latest_markers_detected = 0

        max_markers = (
            self.get_parameter('max_markers_clean')
            .get_parameter_value().integer_value)

        if max_markers > 0:
            self.alpha_cam = max(0.0, min(1.0,
                self.latest_markers_detected / max_markers))
        else:
            self.alpha_cam = 0.0

    def ultrasonic_callback(self, msg):
        """Update sliding window and compute ultrasonic confidence."""
        now = self.get_clock().now().nanoseconds / 1e9

        if len(msg.ranges) > 0:
            range_val = msg.ranges[0]
        else:
            return

        if range_val < msg.range_max:
            self.range_buffer.append((now, range_val))

        window_dur = (
            self.get_parameter('window_duration')
            .get_parameter_value().double_value)
        while self.range_buffer and (now - self.range_buffer[0][0]) > window_dur:
            self.range_buffer.popleft()

        if len(self.range_buffer) >= 3:
            readings = np.array([r for _, r in self.range_buffer])
            variance = np.var(readings)
            k = (self.get_parameter('variance_k')
                 .get_parameter_value().double_value)
            self.alpha_ult = float(np.exp(-k * variance))
        else:
            self.alpha_ult = 1.0

    def publish_confidence(self):
        """Publish both confidence values at 10 Hz."""
        cam_msg = Float64()
        cam_msg.data = self.alpha_cam
        self.cam_conf_pub.publish(cam_msg)

        ult_msg = Float64()
        ult_msg.data = self.alpha_ult
        self.ult_conf_pub.publish(ult_msg)

        if not hasattr(self, '_log_counter'):
            self._log_counter = 0
        self._log_counter += 1
        if self._log_counter % 20 == 0:
            self.get_logger().info(
                f'Confidence — '
                f'cam: {self.alpha_cam:.3f} (markers: {self.latest_markers_detected}) | '
                f'ult: {self.alpha_ult:.3f}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = ConfidenceMonitorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()