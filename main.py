from pathlib import Path

import cv2
import numpy as np


image_path = Path("images/sample.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = gray_image.shape

# Odřízneme spodní černý pruh / watermark oblast
roi_height = int(height * 0.88)
roi = gray_image[:roi_height, :]

# Zvýraznění lokálního kontrastu
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(roi)

# Potlačení šumu při zachování hran
blurred = cv2.medianBlur(
    enhanced,
    9
)

# Hledání kruhových kandidátů
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=70,
    param1=100,
    param2=33,
    minRadius=20,
    maxRadius=55
)

result_image = image.copy()

accepted_circles = []

if circles is not None:
    circles = np.round(circles[0]).astype(int)

    for x, y, radius in circles:
        inner_mask = np.zeros_like(roi)
        ring_mask = np.zeros_like(roi)

        # Oblast uvnitř kandidátního kruhu
        cv2.circle(
            inner_mask,
            (x, y),
            radius,
            255,
            -1
        )

        # Vnější kruh pro okolí objektu
        cv2.circle(
            ring_mask,
            (x, y),
            radius + 15,
            255,
            -1
        )

        # Vyřízneme střed -> zůstane pouze prstenec kolem objektu
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

        print(
            f"x={x}, y={y}, r={radius}, "
            f"inside={inside_brightness:.1f}, "
            f"around={surrounding_brightness:.1f}, "
            f"contrast={contrast:.1f}"
        )

        # Kandidát musí být světlý
        # a zároveň světlejší než své bezprostřední okolí
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

for index, (
    x,
    y,
    radius,
    inside_brightness,
    surrounding_brightness,
    contrast
) in enumerate(accepted_circles, start=1):
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

print(f"Detected circles: {len(accepted_circles)}")

cv2.imwrite(
    "images/enhanced.jpg",
    enhanced
)

cv2.imwrite(
    "images/circles_local_contrast.jpg",
    result_image
)