from datetime import datetime, timezone
from pathlib import Path

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image as PilImage
import pytesseract
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


OCR_TOPIC = "/camera/image_raw"
OBJECT_LABEL_TOPIC = "/object_label"
TESSERACT_CONFIG = "--psm 6"


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


def reshape_interleaved_image(msg: Image, channels: int) -> np.ndarray:
    expected_row_bytes = msg.width * channels
    step = msg.step or expected_row_bytes

    if step < expected_row_bytes:
        raise ValueError(
            f"Image step {step} is smaller than expected row width {expected_row_bytes} for {msg.encoding!r}"
        )

    buffer = np.frombuffer(msg.data, dtype=np.uint8)
    expected_buffer_size = step * msg.height
    if buffer.size < expected_buffer_size:
        raise ValueError(
            f"Image buffer has {buffer.size} bytes but {expected_buffer_size} are required for {msg.encoding!r}"
        )

    row_major = buffer[:expected_buffer_size].reshape((msg.height, step))
    trimmed = row_major[:, :expected_row_bytes]
    return trimmed.reshape((msg.height, msg.width, channels))


def image_message_to_grayscale(msg: Image) -> PilImage.Image:
    encoding = msg.encoding.lower()

    if encoding == "bgr8":
        cv_image = reshape_interleaved_image(msg, 3)
        grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    elif encoding in {"rgb8", "8uc3"}:
        cv_image = reshape_interleaved_image(msg, 3)
        grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    elif encoding in {"rgba8", "8uc4"}:
        cv_image = reshape_interleaved_image(msg, 4)
        grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2GRAY)
    elif encoding == "bgra8":
        cv_image = reshape_interleaved_image(msg, 4)
        grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2GRAY)
    elif encoding in {"mono8", "8uc1"}:
        cv_image = reshape_interleaved_image(msg, 1)
        grayscale_image = cv_image[:, :, 0]
    elif encoding in {"yuv422_yuy2", "yuyv", "yuv422"}:
        cv_image = reshape_interleaved_image(msg, 2)
        bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_YUV2BGR_YUYV)
        grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")

    return PilImage.fromarray(grayscale_image)


class OcrNode(Node):
    def __init__(self) -> None:
        super().__init__("ocr_node")

        load_env_file()

        self.object_label_publisher = self.create_publisher(String, OBJECT_LABEL_TOPIC, 10)
        self.image_subscription = self.create_subscription(Image, OCR_TOPIC, self.image_callback, 10)

        self.get_logger().info(
            f"Listening on {OCR_TOPIC} and publishing OCR text to {OBJECT_LABEL_TOPIC}"
        )

    def image_callback(self, msg: Image) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            grayscale_image = image_message_to_grayscale(msg)
            raw_text = pytesseract.image_to_string(grayscale_image, config=TESSERACT_CONFIG)
        except pytesseract.TesseractNotFoundError as exc:
            self.get_logger().error(f"Tesseract OCR binary not found at {timestamp}: {exc}")
            return
        except Exception as exc:
            self.get_logger().error(f"Failed to extract OCR text at {timestamp}: {exc}")
            return

        published_text = raw_text.strip()
        self.get_logger().info(f"OCR raw_text timestamp={timestamp} text={raw_text!r}")

        if not published_text:
            self.get_logger().info(f"OCR publish skipped timestamp={timestamp} text='' reason=empty_result")
            return

        self.object_label_publisher.publish(String(data=published_text))
        self.get_logger().info(f"OCR published timestamp={timestamp} text={published_text!r}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OcrNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()