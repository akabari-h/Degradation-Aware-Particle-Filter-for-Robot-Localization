"""
AMCL Baseline Launch
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('rsn_p')
    amcl_params = os.path.join(pkg_share, 'config', 'amcl_params.yaml')
    map_yaml = os.path.join(pkg_share, 'config', 'map.yaml')

    # Map server — serves the occupancy grid to AMCL
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            amcl_params,
            {'yaml_filename': map_yaml},
        ],
    )

    # AMCL — Adaptive Monte Carlo Localization
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[amcl_params],
    )

    # Lifecycle manager — activates map_server and AMCL
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server', 'amcl']},
        ],
    )

    return LaunchDescription([
        map_server,
        amcl,
        lifecycle_manager,
    ])