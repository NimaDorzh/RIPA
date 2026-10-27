from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


TEST_IMAGES_DIRECTORY = Path.home() / "robotics_ws" / "test_images"
CAMERA_TOPIC = "/camera/image_raw"
OBJECT_LABEL_TOPIC = "/object_label"
RESPONSE_TIMEOUT_SECONDS = 5.0


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
class OcrTestResult:
    image_file: str
    ocr_text_extracted: str
    published_to_object_label: bool

    @property
    def result_label(self) -> str:
        return "published" if self.published_to_object_label else "timeout"


class OcrTestNode(Node):
    def __init__(self) -> None:
        super().__init__("ocr_test")

        load_env_file()

        self.image_publisher = self.create_publisher(Image, CAMERA_TOPIC, 10)
        self.object_label_subscription = self.create_subscription(
            String,
            OBJECT_LABEL_TOPIC,
            self.object_label_callback,
            10,
        )

        self.pending_response: str | None = None
        self.results: list[OcrTestResult] = []
        self.image_paths = sorted(TEST_IMAGES_DIRECTORY.glob("*.png"))

        self.get_logger().info(
            f"Prepared {len(self.image_paths)} test images from {TEST_IMAGES_DIRECTORY}"
        )

    def object_label_callback(self, msg: String) -> None:
        if self.pending_response is None:
            self.pending_response = msg.data

    def pil_to_image_message(self, image_path: Path) -> Image:
        with PilImage.open(image_path) as image:
            rgb_image = image.convert("RGB")
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = image_path.stem
            msg.height = rgb_image.height
            msg.width = rgb_image.width
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = rgb_image.width * 3
            msg.data = rgb_image.tobytes()
            return msg

    def run(self) -> None:
        if not self.image_paths:
            self.get_logger().warning(f"No PNG files found in {TEST_IMAGES_DIRECTORY}")
            print(self.render_summary_table(), flush=True)
            return

        for image_path in self.image_paths:
            self.pending_response = None
            image_message = self.pil_to_image_message(image_path)
            self.image_publisher.publish(image_message)
            self.get_logger().info(f"Published test image {image_path.name} to {CAMERA_TOPIC}")

            started_ns = self.get_clock().now().nanoseconds
            received_text = ""
            published = False

            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)

                if self.pending_response is not None:
                    received_text = self.pending_response
                    published = True
                    break

                elapsed_seconds = (self.get_clock().now().nanoseconds - started_ns) / 1e9
                if elapsed_seconds >= RESPONSE_TIMEOUT_SECONDS:
                    break

            self.results.append(
                OcrTestResult(
                    image_file=image_path.name,
                    ocr_text_extracted=received_text,
                    published_to_object_label=published,
                )
            )

            if published:
                self.get_logger().info(
                    f"Received {OBJECT_LABEL_TOPIC} text for {image_path.name}: {received_text!r}"
                )
            else:
                self.get_logger().warning(
                    f"Timed out waiting for {OBJECT_LABEL_TOPIC} after publishing {image_path.name}"
                )

        print(self.render_summary_table(), flush=True)

    def render_summary_table(self) -> str:
        rows = [("image", "ocr_text", "result")]
        rows.extend(
            (result.image_file, result.ocr_text_extracted or "", result.result_label)
            for result in self.results
        )

        widths = [max(len(str(row[index])) for row in rows) for index in range(3)]
        lines = []
        for row_index, row in enumerate(rows):
            lines.append(
                " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))
            )
            if row_index == 0:
                lines.append("-+-".join("-" * width for width in widths))
        return "\n".join(lines)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OcrTestNode()

    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()