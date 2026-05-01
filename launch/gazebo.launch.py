from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import xacro
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('rsn_p')
    xacro_file = os.path.join(pkg_share, 'urdf', 'rover.urdf.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'indoor_arena.sdf')
    models_dir = os.path.join(pkg_share, 'models')

    robot_description = xacro.process_file(xacro_file).toxml()

    # Tell Gazebo where to find ArUco marker models AND robot meshes
    set_model_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.dirname(pkg_share) + ':' + models_dir
    )

    # Launch Gazebo with custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': f'-r {world_file}'}.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'robot_description': robot_description},
            {'use_sim_time': True},
        ],
    )

    # Spawn robot at center
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'rover',
            '-topic', 'robot_description',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.1',
        ],
        output='screen',
    )

    # Bridge Gazebo topics to ROS 2
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/camera@sensor_msgs/msg/Image[gz.msgs.Image',
            '/ultrasonic@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/model/rover/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose',
        ],
        output='screen',
    )

    # Odom TF publisher (odom -> base_footprint)
    odom_tf = Node(
        package='rsn_p',
        executable='odom_tf_publisher.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Camera degradation node
    camera_degradation = Node(
        package='rsn_p',
        executable='camera_degradation_node.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Ultrasonic degradation node
    ultrasonic_degradation = Node(
        package='rsn_p',
        executable='ultrasonic_degradation_node.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # LiDAR degradation node (for AMCL baseline comparison)
    lidar_degradation = Node(
        package='rsn_p',
        executable='lidar_degradation_node.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Confidence monitor node
    confidence_monitor = Node(
        package='rsn_p',
        executable='confidence_monitor_node.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # Particle filter node (DA-PF)
    particle_filter = Node(
        package='rsn_p',
        executable='particle_filter_node.py',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    return LaunchDescription([
        set_model_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        odom_tf,
        camera_degradation,
        ultrasonic_degradation,
        lidar_degradation,
        confidence_monitor,
        particle_filter,
    ])