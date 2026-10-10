from pathlib import Path

import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image


def prepare_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, _ = gray_image.shape
    roi_height = int(height * 0.88)

    roi = gray_image[:roi_height, :]

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(roi)

    blurred = cv2.medianBlur(
        enhanced,
        9
    )

    return roi, blurred


def detect_circles(image: np.ndarray) -> np.ndarray | None:
    return cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=70,
        param1=100,
        param2=33,
        minRadius=20,
        maxRadius=55
    )


def calculate_circle_brightness(
    roi: np.ndarray,
    x: int,
    y: int,
    radius: int
) -> tuple[float, float, float]:
    inner_mask = np.zeros_like(roi)
    ring_mask = np.zeros_like(roi)

    cv2.circle(
        inner_mask,
        (x, y),
        radius,
        255,
        -1
    )

    cv2.circle(
        ring_mask,
        (x, y),
        radius + 15,
        255,
        -1
    )

    cv2.circle(
        ring_mask,
        (x, y),
        radius + 3,
        0,
        -1
    )

    inside_brightness = cv2.mean(
        roi,
        mask=inner_mask
    )[0]

    surrounding_brightness = cv2.mean(
        roi,
        mask=ring_mask
    )[0]

    contrast = inside_brightness - surrounding_brightness

    return (
        inside_brightness,
        surrounding_brightness,
        contrast
    )


def filter_candidates(
    roi: np.ndarray,
    circles: np.ndarray | None
) -> list[tuple[int, int, int, float, float, float]]:
    if circles is None:
        return []

    detected_circles = np.round(circles[0]).astype(int)

    accepted_circles = []

    for x, y, radius in detected_circles:
        inside_brightness, surrounding_brightness, contrast = (
            calculate_circle_brightness(
                roi,
                x,
                y,
                radius
            )
        )

        print(
            f"x={x}, y={y}, r={radius}, "
            f"inside={inside_brightness:.1f}, "
            f"around={surrounding_brightness:.1f}, "
            f"contrast={contrast:.1f}"
        )

        if inside_brightness >= 150 and contrast >= 10:
            accepted_circles.append(
                (
                    x,
                    y,
                    radius,
                    inside_brightness,
                    surrounding_brightness,
                    contrast
                )
            )

    return accepted_circles


def draw_results(
    image: np.ndarray,
    circles: list[tuple[int, int, int, float, float, float]]
) -> np.ndarray:
    result_image = image.copy()

    for index, (
        x,
        y,
        radius,
        inside_brightness,
        surrounding_brightness,
        contrast
    ) in enumerate(circles, start=1):
        cv2.circle(
            result_image,
            (x, y),
            radius,
            (0, 255, 0),
            3
        )

        cv2.circle(
            result_image,
            (x, y),
            3,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            result_image,
            str(index),
            (x + radius, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print(
            f"Accepted #{index}: "
            f"x={x}, y={y}, r={radius}, "
            f"inside={inside_brightness:.1f}, "
            f"around={surrounding_brightness:.1f}, "
            f"contrast={contrast:.1f}"
        )

    return result_image


def save_result(
    output_path: Path,
    image: np.ndarray
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(output_path),
        image
    )