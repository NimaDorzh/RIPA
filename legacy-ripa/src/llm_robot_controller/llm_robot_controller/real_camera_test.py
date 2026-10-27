import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from geometry_msgs.msg import TwistStamped
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from llm_robot_controller.results_paths import CSV_RESULTS_DIR


OBJECT_LABEL_TOPIC = "/object_label"
OBJECT_LABEL_SAFE_TOPIC = "/object_label_safe"
FIREWALL_BLOCKED_TOPIC = "/firewall_blocked"
CMD_VEL_TOPIC = "/cmd_vel"
CONTROLLER_TIMEOUT_SECONDS = 10.0


def load_env_file() -> None:
    checked_paths = set()

    for base_path in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (base_path, *base_path.parents):
            env_path = directory / ".env"
            if env_path in checked_paths or not env_path.is_file():
                continue

            checked_paths.add(env_path)
            load_dotenv(env_path, override=False)
            return


@dataclass
class CameraObservation:
    timestamp: str
    created_at_seconds: float
    ocr_text: str
    firewall_decision: str = "PENDING"
    controller_action: str = "PENDING"


class RealCameraTestNode(Node):
    def __init__(self) -> None:
        super().__init__("real_camera_test")

        load_env_file()

        CSV_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_path = CSV_RESULTS_DIR / f"camera_test_{timestamp}.csv"
        self.results_handle = self.results_path.open("w", newline="", encoding="utf-8")
        self.results_writer = csv.DictWriter(
            self.results_handle,
            fieldnames=["timestamp", "ocr_text", "firewall_decision", "controller_action"],
        )
        self.results_writer.writeheader()

        self.pending_observations: list[CameraObservation] = []

        self.create_subscription(String, OBJECT_LABEL_TOPIC, self.object_label_callback, 10)
        self.create_subscription(String, FIREWALL_BLOCKED_TOPIC, self.firewall_blocked_callback, 10)
        self.create_subscription(String, OBJECT_LABEL_SAFE_TOPIC, self.object_label_safe_callback, 10)
        self.create_subscription(TwistStamped, CMD_VEL_TOPIC, self.cmd_vel_callback, 10)
        self.timeout_timer = self.create_timer(1.0, self.flush_timed_out_observations)

        self.get_logger().info(f"Recording real camera results to {self.results_path}")

    def object_label_callback(self, msg: String) -> None:
        ocr_text = msg.data.strip()
        if not ocr_text:
            return

        observation = CameraObservation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            created_at_seconds=self.get_clock().now().nanoseconds / 1e9,
            ocr_text=ocr_text,
        )
        self.pending_observations.append(observation)
        print(f'[CAMERA] OCR received: "{ocr_text}"', flush=True)

    def firewall_blocked_callback(self, msg: String) -> None:
        observation = self.find_pending_observation(msg.data, allow_pending_only=True)
        if observation is None:
            observation = self.create_recovered_observation(msg.data)

        observation.firewall_decision = "BLOCK"
        observation.controller_action = "BLOCKED"
        print("[FIREWALL] Decision: BLOCK", flush=True)
        self.persist_observation(observation)

    def object_label_safe_callback(self, msg: String) -> None:
        observation = self.find_pending_observation(msg.data, allow_pending_only=True)
        if observation is None:
            observation = self.create_recovered_observation(msg.data)

        observation.firewall_decision = "ALLOW"
        print("[FIREWALL] Decision: ALLOW", flush=True)

    def cmd_vel_callback(self, msg: TwistStamped) -> None:
        observation = self.find_pending_observation(require_allow=True)
        if observation is None:
            return

        observation.controller_action = self.infer_controller_action(msg)
        self.persist_observation(observation)

    def flush_timed_out_observations(self) -> None:
        now_seconds = self.get_clock().now().nanoseconds / 1e9
        timed_out = [
            observation
            for observation in self.pending_observations
            if observation.firewall_decision == "ALLOW"
            and observation.controller_action == "PENDING"
            and now_seconds - observation.created_at_seconds >= CONTROLLER_TIMEOUT_SECONDS
        ]

        for observation in timed_out:
            observation.controller_action = "NO_ACTION"
            self.persist_observation(observation)

    def find_pending_observation(
        self,
        ocr_text: str | None = None,
        *,
        allow_pending_only: bool = False,
        require_allow: bool = False,
    ) -> CameraObservation | None:
        normalized_text = ocr_text.strip() if ocr_text is not None else None

        for observation in self.pending_observations:
            if normalized_text is not None and observation.ocr_text != normalized_text:
                continue
            if allow_pending_only and observation.firewall_decision != "PENDING":
                continue
            if require_allow and observation.firewall_decision != "ALLOW":
                continue
            if observation.controller_action != "PENDING":
                continue
            return observation

        return None

    def create_recovered_observation(self, ocr_text: str) -> CameraObservation:
        observation = CameraObservation(
            timestamp=datetime.now(timezone.utc).isoformat(),
            created_at_seconds=self.get_clock().now().nanoseconds / 1e9,
            ocr_text=ocr_text.strip(),
        )
        self.pending_observations.append(observation)
        return observation

    def persist_observation(self, observation: CameraObservation) -> None:
        self.results_writer.writerow(
            {
                "timestamp": observation.timestamp,
                "ocr_text": observation.ocr_text,
                "firewall_decision": observation.firewall_decision,
                "controller_action": observation.controller_action,
            }
        )
        self.results_handle.flush()
        if observation in self.pending_observations:
            self.pending_observations.remove(observation)

    @staticmethod
    def infer_controller_action(msg: TwistStamped) -> str:
        linear_x = msg.twist.linear.x
        if linear_x > 0.0:
            return "MOVE_ZONE_A"
        if linear_x < 0.0:
            return "MOVE_ZONE_B"
        return "STOP"

    def close(self) -> None:
        for observation in list(self.pending_observations):
            if observation.firewall_decision == "PENDING":
                observation.firewall_decision = "UNKNOWN"
            if observation.controller_action == "PENDING":
                observation.controller_action = "NO_ACTION"
            self.persist_observation(observation)

        self.results_handle.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RealCameraTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()