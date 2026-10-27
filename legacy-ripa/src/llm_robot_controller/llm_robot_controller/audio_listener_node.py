"""ROS 2 audio listener node for reproducible .wav-based audio injection tests.

This node intentionally does not support live microphone capture. It only
transcribes a provided .wav file so experiments remain deterministic and easy
to replay across runs.
"""

from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from std_msgs.msg import String
import whisper


class AudioListenerNode(Node):
    def __init__(self) -> None:
        super().__init__("audio_listener")

        self.declare_parameter("audio_file", "")
        self.declare_parameter("publish_interval", 3.0)

        self.audio_file = self.get_parameter("audio_file").get_parameter_value().string_value
        self.publish_interval = (
            self.get_parameter("publish_interval").get_parameter_value().double_value
        )

        qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # The llm_controller_node subscriber must also use TRANSIENT_LOCAL QoS.
        self.publisher = self.create_publisher(String, "/object_label", qos)

        self.transcription = self._transcribe_audio_once()
        self.get_logger().info(f"Transcribed: {self.transcription}")

        self.timer = self.create_timer(self.publish_interval, self._publish_transcription)

    def _transcribe_audio_once(self) -> str:
        if not self.audio_file.strip():
            self.get_logger().warning("Parameter 'audio_file' is empty; publishing an empty transcription")
            return ""

        audio_path = Path(self.audio_file).expanduser()
        try:
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            model = whisper.load_model("base")
            result = model.transcribe(str(audio_path), fp16=False)
            return result.get("text", "").strip()
        except FileNotFoundError as exc:
            self.get_logger().error(str(exc))
            return ""
        except Exception as exc:
            self.get_logger().error(f"Failed to transcribe '{audio_path}': {exc}")
            return ""

    def _publish_transcription(self) -> None:
        publisher_infos = self.get_publishers_info_by_topic("/object_label")
        subscriber_count = self.count_subscribers("/object_label")
        if not publisher_infos or subscriber_count < 1:
            self.get_logger().warning("No subscribers detected on /object_label; skipping publish")
            return

        message = String()
        message.data = self.transcription
        self.publisher.publish(message)
        self.get_logger().info(f"Published to /object_label: {self.transcription}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AudioListenerNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()