from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PilImage
from PIL import ImageOps
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


def image_message_to_pil(msg: Image) -> PilImage.Image:
    encoding = msg.encoding.lower()
    mode_map = {
        "rgb8": ("RGB", "RGB", 3),
        "bgr8": ("RGB", "BGR", 3),
        "rgba8": ("RGBA", "RGBA", 4),
        "bgra8": ("RGBA", "BGRA", 4),
        "mono8": ("L", "L", 1),
        "8uc1": ("L", "L", 1),
        "8uc3": ("RGB", "RGB", 3),
        "8uc4": ("RGBA", "RGBA", 4),
    }

    if encoding not in mode_map:
        raise ValueError(f"Unsupported image encoding: {msg.encoding!r}")

    mode, raw_mode, channels = mode_map[encoding]
    expected_step = msg.width * channels
    step = msg.step or expected_step
    image = PilImage.frombuffer(
        mode,
        (msg.width, msg.height),
        bytes(msg.data),
        "raw",
        raw_mode,
        step,
        1,
    )

    if step < expected_step:
        raise ValueError(
            f"Image step {step} is smaller than expected row width {expected_step} for {msg.encoding!r}"
        )

    return image.copy()


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
            image = image_message_to_pil(msg)
            grayscale_image = ImageOps.grayscale(image)
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