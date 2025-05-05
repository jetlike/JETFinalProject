# ros_pkg/launch/multimodallaunch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ros_pkg',
            executable='facial_recognition_node',
            name='facial_recognition_node',
            output='screen',
        ),
        Node(
            package='ros_pkg',
            executable='listener_node',
            name='listener_node',
            output='screen',
        ),
        Node(
            package='ros_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
        ),
        Node(
            package='ros_pkg',
            executable='llm_node',
            name='llm_node',
            output='screen',
        ),
    ])