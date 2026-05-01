#!/usr/bin/env python3
"""
LiDAR Degradation Node
=======================
Applies the same style of degradation as the ultrasonic node
to the 2D LiDAR used by AMCL, ensuring a fair comparison.


"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class LidarDegradationNode(Node):

    def __init__(self):
        super().__init__('lidar_degradation')

        self.declare_parameter('noise_stddev', 0.0)
        self.declare_parameter('dropout_rate', 0.0)

        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        self.publisher = self.create_publisher(
            LaserScan, '/scan/degraded', 10
        )

        self.get_logger().info('LiDAR degradation node started')
        self.get_logger().info('  Subscribes: /scan → Publishes: /scan/degraded')

    def scan_callback(self, msg):
        noise_std = self.get_parameter('noise_stddev').get_parameter_value().double_value
        dropout = self.get_parameter('dropout_rate').get_parameter_value().double_value

        degraded_msg = LaserScan()
        degraded_msg.header = msg.header
        # Override Gazebo frame name to match TF tree
        degraded_msg.header.frame_id = 'lidar_link'
        degraded_msg.angle_min = msg.angle_min
        degraded_msg.angle_max = msg.angle_max
        degraded_msg.angle_increment = msg.angle_increment
        degraded_msg.time_increment = msg.time_increment
        degraded_msg.scan_time = msg.scan_time
        degraded_msg.range_min = msg.range_min
        degraded_msg.range_max = msg.range_max

        ranges = np.array(msg.ranges, dtype=np.float32)

        if noise_std > 0.0:
            ranges += np.random.normal(0.0, noise_std, size=ranges.shape).astype(np.float32)

        if dropout > 0.0:
            mask = np.random.random(size=ranges.shape) < dropout
            ranges[mask] = msg.range_max

        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        degraded_msg.ranges = ranges.tolist()
        degraded_msg.intensities = list(msg.intensities) if msg.intensities else []

        self.publisher.publish(degraded_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarDegradationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()