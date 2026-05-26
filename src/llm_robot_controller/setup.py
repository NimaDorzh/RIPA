from setuptools import find_packages
from setuptools import setup


package_name = "llm_robot_controller"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "openai>=1.0.0"],
    zip_safe=True,
    maintainer="amin",
    maintainer_email="amin@example.com",
    description="ROS 2 GPT-4o controller that converts object labels into cmd_vel commands.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "controller_node = llm_robot_controller.controller_node:main",
        ],
    },
)