from setuptools import setup, find_packages

setup(
    name='ros_pkg',
    version='0.1.0',
    packages=find_packages(),  # will find ros_pkg/
    install_requires=[
        'setuptools',
    ],
    entry_points={
        'console_scripts': [
            'facial_recognition_node = ros_pkg.facial_recognition_node:main',
            'listener_node          = ros_pkg.listener_node:main',
            'vision_node            = ros_pkg.vision_node:main',
            'llm_node               = ros_pkg.llm_node:main',
        ],
    },
)
