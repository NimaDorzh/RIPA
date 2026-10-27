import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


SCAN_TOPIC = "/scan"
SENSOR_CONTEXT_TOPIC = "/sensor_context"
SECTOR_LABELS = [
    "front",
    "front-right",
    "right",
    "rear-right",
    "rear",
    "rear-left",
    "left",
    "front-left",
]
SECTOR_CENTERS = {
    "front": 0.0,
    "front-right": -math.pi / 4.0,
    "right": -math.pi / 2.0,
    "rear-right": -3.0 * math.pi / 4.0,
    "rear": math.pi,
    "rear-left": 3.0 * math.pi / 4.0,
    "left": math.pi / 2.0,
    "front-left": math.pi / 4.0,
}
SECTOR_WIDTH = math.pi / 4.0


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def format_sector_description(distance: float | None) -> str:
    if distance is None or distance > 2.0:
        return "clear"

    if distance <= 0.5:
        return f"obstacle at {distance:.2f}m (CLOSE!)"

    if distance <= 1.0:
        return f"obstacle at {distance:.2f}m (near)"

    return f"obstacle at {distance:.2f}m"


class SensorContextNode(Node):
    def __init__(self) -> None:
        super().__init__("sensor_context_node")

        scan_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        context_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.context_publisher = self.create_publisher(String, SENSOR_CONTEXT_TOPIC, context_qos)
        self.scan_subscription = self.create_subscription(
            LaserScan,
            SCAN_TOPIC,
            self.scan_callback,
            scan_qos,
        )

        self.get_logger().info("sensor_context_node started — listening on /scan")

    def scan_callback(self, msg: LaserScan) -> None:
        sector_minimums = {label: None for label in SECTOR_LABELS}

        for index, distance in enumerate(msg.ranges):
            if math.isinf(distance) or math.isnan(distance):
                continue

            angle = wrap_angle(msg.angle_min + index * msg.angle_increment)
            sector_label = self._sector_for_angle(angle)
            current_minimum = sector_minimums[sector_label]

            if current_minimum is None or distance < current_minimum:
                sector_minimums[sector_label] = distance

        descriptions = [
            f"{label}: {format_sector_description(sector_minimums[label])}"
            for label in SECTOR_LABELS
        ]
        context = f"LiDAR scan: {', '.join(descriptions)}."

        self.context_publisher.publish(String(data=context))
        self.get_logger().debug(f"Published context: {context}")

    def _sector_for_angle(self, angle: float) -> str:
        for label in SECTOR_LABELS:
            center = SECTOR_CENTERS[label]
            if abs(wrap_angle(angle - center)) < SECTOR_WIDTH / 2.0:
                return label

        return "rear"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorContextNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()