import os
from pathlib import Path
from typing import Optional

from geometry_msgs.msg import TwistStamped
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

LLM_TARGETS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "",
        "model": "gpt-4o",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
    },
}


def load_env_file() -> None:
    checked_paths = set()

    for base_path in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (base_path, *base_path.parents):
            env_path = directory / ".env"
            if env_path in checked_paths or not env_path.is_file():
                continue

            checked_paths.add(env_path)

            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
            return


class LlmRobotController(Node):
    def __init__(self) -> None:
        super().__init__("llm_robot_controller")

        load_env_file()

        default_target = os.getenv("TARGET_LLM", "deepseek").strip().lower()
        self.declare_parameter("target_llm", default_target)
        self.target_llm = self.get_parameter("target_llm").get_parameter_value().string_value.lower()

        if self.target_llm not in LLM_TARGETS:
            supported_targets = ", ".join(sorted(LLM_TARGETS))
            raise RuntimeError(
                f"Unsupported target_llm '{self.target_llm}'. Supported values: {supported_targets}"
            )

        target_config = LLM_TARGETS[self.target_llm]
        default_model = os.getenv("TARGET_LLM_MODEL", target_config["model"])
        default_base_url = os.getenv("TARGET_LLM_BASE_URL", target_config["base_url"])

        self.declare_parameter("llm_model", default_model)
        self.model = self.get_parameter("llm_model").get_parameter_value().string_value

        self.declare_parameter("llm_base_url", default_base_url)
        self.base_url = self.get_parameter("llm_base_url").get_parameter_value().string_value

        api_key = os.getenv("TARGET_LLM_API_KEY") or os.getenv(target_config["api_key_env"])
        if not api_key:
            raise RuntimeError(
                f"{target_config['api_key_env']} is not set. Add it to .env or export it in the environment"
            )

        if self.base_url:
            self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=api_key)
        self.cmd_vel_publisher = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.object_label_subscription = self.create_subscription(
            String,
            "/object_label",
            self.object_label_callback,
            10,
        )

        self.get_logger().info(
            f"Listening on /object_label and publishing to /cmd_vel with target {self.target_llm}, model {self.model}"
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
            self.get_logger().error(f"LLM API request failed: {exc}")
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
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()

        if command == "MOVE_ZONE_A":
            msg.twist.linear.x = 0.5
        elif command == "MOVE_ZONE_B":
            msg.twist.linear.x = -0.5
        else:
            msg.twist.linear.x = 0.0

        msg.twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(msg)


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