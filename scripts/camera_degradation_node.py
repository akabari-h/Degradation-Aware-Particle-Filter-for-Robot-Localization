#!/usr/bin/env python3
"""
Camera Degradation Node

Subscribes to the clean camera image, applies fog/smoke degradation,
and republishes the degraded image.

"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraDegradationNode(Node):

    def __init__(self):
        super().__init__('camera_degradation')

        # Declare tunable parameters with defaults (clean = no degradation)
        self.declare_parameter('blur_kernel_size', 1)
        self.declare_parameter('contrast_alpha', 1.0)

        # CV bridge for converting ROS Image <-> OpenCV
        self.bridge = CvBridge()

        # Subscribe to clean camera topic from Gazebo bridge
        self.subscription = self.create_subscription(
            Image,
            '/camera',
            self.image_callback,
            10
        )

        # Publish degraded image
        self.publisher = self.create_publisher(
            Image,
            '/camera/degraded',
            10
        )

        self.get_logger().info('Camera degradation node started')
        self.get_logger().info('  Params: blur_kernel_size=1, contrast_alpha=1.0 (clean)')
        self.get_logger().info('  Tune: ros2 param set /camera_degradation blur_kernel_size 21')

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Read current parameter values (allows runtime tuning)
        blur_k = self.get_parameter('blur_kernel_size').get_parameter_value().integer_value
        contrast = self.get_parameter('contrast_alpha').get_parameter_value().double_value

        # ---- Apply degradation ----

        degraded = cv_image.copy()

        # Step 1: Gaussian blur (simulates fog scattering)
        # Kernel must be odd and >= 1
        if blur_k > 1:
            if blur_k % 2 == 0:
                blur_k += 1  # force odd
            degraded = cv2.GaussianBlur(degraded, (blur_k, blur_k), 0)

        # Step 2: Contrast reduction (simulates visibility loss)
        # Blend image toward its mean pixel value
        # degraded = alpha * image + (1 - alpha) * mean_gray
        if contrast < 1.0:
            mean_val = np.mean(degraded)
            gray_image = np.full_like(degraded, mean_val, dtype=np.uint8)
            degraded = cv2.addWeighted(
                degraded, contrast,
                gray_image, 1.0 - contrast,
                0
            )

        # Convert back to ROS Image and publish
        try:
            degraded_msg = self.bridge.cv2_to_imgmsg(degraded, encoding='bgr8')
            degraded_msg.header = msg.header  # preserve timestamp
            self.publisher.publish(degraded_msg)
        except Exception as e:
            self.get_logger().error(f'CV Bridge publish error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = CameraDegradationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()