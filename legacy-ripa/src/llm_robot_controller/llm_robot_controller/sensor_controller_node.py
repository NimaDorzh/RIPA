import os
from pathlib import Path

from dotenv import load_dotenv
from geometry_msgs.msg import TwistStamped
from openai import OpenAI
from rclpy.node import Node
import rclpy
from rclpy.logging import get_logger
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String


DEFAULT_TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
STOP_ACTION = "STOP"
VALID_ACTIONS = {"MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", STOP_ACTION}
ACTION_TO_TWIST = {
    "MOVE_FORWARD": (0.15, 0.0),
    "TURN_LEFT": (0.0, 0.5),
    "TURN_RIGHT": (0.0, -0.5),
    STOP_ACTION: (0.0, 0.0),
}


def load_workspace_env() -> None:
    load_dotenv(Path.home() / "robotics_ws" / ".env", override=False)


class SensorControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("sensor_controller_node")

        load_workspace_env()

        self.declare_parameter("llm_provider", "deepseek")
        self.declare_parameter("together_model", DEFAULT_TOGETHER_MODEL)

        self.llm_provider = self.get_parameter("llm_provider").get_parameter_value().string_value.strip().lower()
        if not self.llm_provider:
            self.llm_provider = "deepseek"

        together_model = self.get_parameter("together_model").get_parameter_value().string_value.strip()
        self.model = together_model or DEFAULT_TOGETHER_MODEL

        if self.llm_provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
            base_url = "https://api.deepseek.com/v1"
            self.model = "deepseek-chat"
        elif self.llm_provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY", "").strip()
            base_url = "https://api.together.ai/v1"
        else:
            raise RuntimeError(
                "Unsupported llm_provider "
                f"'{self.llm_provider}'. Supported values: deepseek, together"
            )

        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider '{self.llm_provider}'. Check ~/robotics_ws/.env"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.cmd_vel_publisher = self.create_publisher(TwistStamped, "/cmd_vel", qos)
        self.sensor_context_subscription = self.create_subscription(
            String,
            "/sensor_context",
            self.sensor_context_callback,
            qos,
        )

        self.get_logger().info(
            "Listening on /sensor_context and publishing to /cmd_vel with provider "
            f"{self.llm_provider}, model {self.model}"
        )

    def sensor_context_callback(self, msg: String) -> None:
        sensor_context = msg.data.strip()
        self.get_logger().info(f"Sensor context received: {sensor_context[:80]}...")

        try:
            response = self.query_llm(sensor_context)
            action = self.parse_action(response)
        except Exception as exc:
            self.get_logger().error(f"LLM API request failed: {exc}")
            response = STOP_ACTION
            action = STOP_ACTION

        self.get_logger().info(f"LLM response: {response} -> publishing {action}")
        self.publish_command(action)

    def query_llm(self, sensor_context: str) -> str:
        system_prompt = f"""You are a robot navigation controller. Your goal is to navigate to zone A.
Current environment state from LiDAR sensors:
{sensor_context}

Based on this sensor data, decide the robot's next action.
If the path forward (front sector) is clear: move forward toward zone A.
If front is blocked (near or CLOSE) but at least one other sector is clear: turn toward the clearest path.
If ALL sectors are blocked (all report near or CLOSE): stop immediately and output STOP.
Respond ONLY with one of: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP. Nothing else."""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=10,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "What is the robot's next action?"},
            ],
        )

        content = response.choices[0].message.content
        return content.strip() if content else STOP_ACTION

    def parse_action(self, response: str) -> str:
        normalized = response.strip().upper()
        if normalized in VALID_ACTIONS:
            return normalized

        self.get_logger().warning(
            f"Unexpected LLM response: '{response}' — defaulting to STOP"
        )
        return STOP_ACTION

    def publish_command(self, action: str) -> None:
        linear_x, angular_z = ACTION_TO_TWIST.get(action, ACTION_TO_TWIST[STOP_ACTION])

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = linear_x
        msg.twist.angular.z = angular_z
        self.cmd_vel_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None

    try:
        node = SensorControllerNode()
        rclpy.spin(node)
    except RuntimeError as exc:
        if rclpy.ok():
            get_logger("sensor_controller_node").error(str(exc))
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()