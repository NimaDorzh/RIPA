import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import LaserScan


SCAN_TOPIC = "/scan"
DEFAULT_VARIANT = "N"
DEFAULT_PUBLISH_RATE = 10.0
NUM_RANGES = 360
RANGE_CLEAR = 3.5
RANGE_BLOCKED = 0.3
RANGE_MIN = 0.12
RANGE_MAX = 3.5
ANGLE_MIN = -math.pi
ANGLE_MAX = math.pi
ANGLE_INCREMENT = 2.0 * math.pi / NUM_RANGES
VALID_VARIANTS = {"N", "A1", "A2", "A3"}


class SensorSpoofNode(Node):
    def __init__(self) -> None:
        super().__init__("sensor_spoof_node")

        self.declare_parameter("variant", DEFAULT_VARIANT)
        self.declare_parameter("publish_rate", DEFAULT_PUBLISH_RATE)

        self.variant = self.get_parameter("variant").get_parameter_value().string_value.upper()
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        self.publish_rate = publish_rate if publish_rate > 0.0 else DEFAULT_PUBLISH_RATE

        if self.variant not in VALID_VARIANTS:
            self.get_logger().warning(
                f"Unknown variant '{self.variant}', falling back to {DEFAULT_VARIANT}"
            )
            self.variant = DEFAULT_VARIANT

        if publish_rate <= 0.0:
            self.get_logger().warning(
                f"Invalid publish_rate '{publish_rate}', falling back to {DEFAULT_PUBLISH_RATE}Hz"
            )

        scan_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_publisher = self.create_publisher(LaserScan, SCAN_TOPIC, scan_qos)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_scan)

        self._front_indices = self._compute_indices_for_angle_window(
            -math.pi / 8.0,
            math.pi / 8.0,
        )

        self.get_logger().info(
            f"sensor_spoof_node started — variant={self.variant}, rate={self.publish_rate}Hz"
        )

    def publish_scan(self) -> None:
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = "base_link"
        scan_msg.angle_min = ANGLE_MIN
        scan_msg.angle_max = ANGLE_MAX
        scan_msg.angle_increment = ANGLE_INCREMENT
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / self.publish_rate
        scan_msg.range_min = RANGE_MIN
        scan_msg.range_max = RANGE_MAX
        scan_msg.ranges = self._build_ranges()
        scan_msg.intensities = []

        self.scan_publisher.publish(scan_msg)
        self.get_logger().debug(f"Published spoofed scan: variant={self.variant}")

    def _build_ranges(self) -> list[float]:
        if self.variant == "A2":
            return [RANGE_BLOCKED] * NUM_RANGES

        if self.variant == "A3":
            # Ghost obstacles on every side/rear sector; front stays clear.
            ranges = [RANGE_BLOCKED] * NUM_RANGES
            for index in self._front_indices:
                ranges[index] = RANGE_CLEAR
            return ranges

        ranges = [RANGE_CLEAR] * NUM_RANGES

        if self.variant == "A1":
            for index in self._front_indices:
                ranges[index] = RANGE_BLOCKED

        return ranges

    def _compute_indices_for_angle_window(
        self,
        window_min: float,
        window_max: float,
    ) -> list[int]:
        indices = []

        for index in range(NUM_RANGES):
            angle = ANGLE_MIN + index * ANGLE_INCREMENT
            if window_min <= angle <= window_max:
                indices.append(index)

        return indices


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorSpoofNode()

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