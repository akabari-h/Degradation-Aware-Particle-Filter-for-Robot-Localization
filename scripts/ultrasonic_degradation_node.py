#!/usr/bin/env python3
"""
Ultrasonic Degradation Node
============================
Subscribes to the clean ultrasonic range topic, adds noise and
missed echoes, and republishes the degraded readings.


"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np


class UltrasonicDegradationNode(Node):

    def __init__(self):
        super().__init__('ultrasonic_degradation')

        # Declare tunable parameters with defaults (clean = no degradation)
        self.declare_parameter('noise_stddev', 0.0)
        self.declare_parameter('dropout_rate', 0.0)

        # Subscribe to clean ultrasonic topic from Gazebo bridge
        self.subscription = self.create_subscription(
            LaserScan,
            '/ultrasonic',
            self.scan_callback,
            10
        )

        # Publish degraded readings
        self.publisher = self.create_publisher(
            LaserScan,
            '/ultrasonic/degraded',
            10
        )

        self.get_logger().info('Ultrasonic degradation node started')
        self.get_logger().info('  Params: noise_stddev=0.0, dropout_rate=0.0 (clean)')
        self.get_logger().info('  Tune: ros2 param set /ultrasonic_degradation noise_stddev 0.3')

    def scan_callback(self, msg):
        # Read current parameter values (allows runtime tuning)
        noise_std = self.get_parameter('noise_stddev').get_parameter_value().double_value
        dropout = self.get_parameter('dropout_rate').get_parameter_value().double_value

        # Copy the message
        degraded_msg = LaserScan()
        degraded_msg.header = msg.header
        degraded_msg.angle_min = msg.angle_min
        degraded_msg.angle_max = msg.angle_max
        degraded_msg.angle_increment = msg.angle_increment
        degraded_msg.time_increment = msg.time_increment
        degraded_msg.scan_time = msg.scan_time
        degraded_msg.range_min = msg.range_min
        degraded_msg.range_max = msg.range_max

        # Convert to numpy for processing
        ranges = np.array(msg.ranges, dtype=np.float32)

        # Step 1: Add Gaussian noise
        if noise_std > 0.0:
            noise = np.random.normal(0.0, noise_std, size=ranges.shape)
            ranges = ranges + noise.astype(np.float32)

        # Step 2: Random dropout (missed echoes → max range)
        if dropout > 0.0:
            dropout_mask = np.random.random(size=ranges.shape) < dropout
            ranges[dropout_mask] = msg.range_max

        # Step 3: Clip to valid sensor bounds
        ranges = np.clip(ranges, msg.range_min, msg.range_max)

        degraded_msg.ranges = ranges.tolist()
        degraded_msg.intensities = list(msg.intensities)

        self.publisher.publish(degraded_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicDegradationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()