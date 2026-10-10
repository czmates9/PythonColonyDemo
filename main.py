from pathlib import Path

import cv2


image_path = Path("images/sample.jpg")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

height, width, channels = image.shape

print(f"Width: {width}")
print(f"Height: {height}")
print(f"Channels: {channels}")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

output_path = Path("images/sample_gray.jpg")
cv2.imwrite(str(output_path), gray_image)

print(f"Grayscale image saved to: {output_path}")