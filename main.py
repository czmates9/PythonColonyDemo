from pathlib import Path

from src.image_processing import (
    detect_circles,
    draw_results,
    filter_candidates,
    load_image,
    prepare_image,
    save_result,
)


IMAGE_PATH = Path("images/sample.jpg")
OUTPUT_PATH = Path("output/circles_local_contrast.jpg")


def main() -> None:
    image = load_image(IMAGE_PATH)

    roi, prepared_image = prepare_image(image)

    circles = detect_circles(prepared_image)

    accepted_circles = filter_candidates(
        roi,
        circles
    )

    result_image = draw_results(
        image,
        accepted_circles
    )

    save_result(
        OUTPUT_PATH,
        result_image
    )

    print(
        f"Detected circles: {len(accepted_circles)}"
    )


if __name__ == "__main__":
    main()