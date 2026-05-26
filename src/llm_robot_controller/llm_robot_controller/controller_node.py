import os
from typing import Optional

from geometry_msgs.msg import Twist
from openai import OpenAI
from openai import OpenAIError
import rclpy
from rclpy.logging import get_logger
from rclpy.node import Node
from std_msgs.msg import String


SYSTEM_PROMPT = (
    "You are a robot controller. Based on the instruction, "
    "return only one command: MOVE_ZONE_A or MOVE_ZONE_B"
)


class LlmRobotController(Node):
    def __init__(self) -> None:
        super().__init__("llm_robot_controller")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.declare_parameter("openai_model", "gpt-4o")
        self.model = self.get_parameter("openai_model").get_parameter_value().string_value

        self.client = OpenAI(api_key=api_key)
        self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)
        self.object_label_subscription = self.create_subscription(
            String,
            "/object_label",
            self.object_label_callback,
            10,
        )

        self.get_logger().info(
            f"Listening on /object_label and publishing to /cmd_vel with model {self.model}"
        )

    def object_label_callback(self, msg: String) -> None:
        label = msg.data.strip()
        if not label:
            self.get_logger().warning("Received empty /object_label message, stopping robot")
            self.publish_command(None)
            return

        self.get_logger().info(f"Received label: {label}")

        try:
            llm_command = self.query_llm(label)
        except OpenAIError as exc:
            self.get_logger().error(f"OpenAI API request failed: {exc}")
            self.publish_command(None)
            return
        except Exception as exc:
            self.get_logger().error(f"Unexpected LLM error: {exc}")
            self.publish_command(None)
            return

        self.get_logger().info(f"LLM returned command: {llm_command or 'UNKNOWN'}")
        self.publish_command(llm_command)

    def query_llm(self, label: str) -> Optional[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": label},
            ],
        )

        content = response.choices[0].message.content
        if not content:
            return None

        normalized = content.strip().upper()
        if normalized in {"MOVE_ZONE_A", "MOVE_ZONE_B"}:
            return normalized
        return None

    def publish_command(self, command: Optional[str]) -> None:
        twist = Twist()

        if command == "MOVE_ZONE_A":
            twist.linear.x = 0.5
        elif command == "MOVE_ZONE_B":
            twist.linear.x = -0.5
        else:
            twist.linear.x = 0.0

        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = LlmRobotController()
        rclpy.spin(node)
    except RuntimeError as exc:
        if rclpy.ok():
            temp_logger = get_logger("llm_robot_controller")
            temp_logger.error(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()