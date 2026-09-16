

import argparse

import cv2
import numpy as np


def to_grayscale(img):
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise(gray, strength=10):
    return cv2.fastNlMeansDenoising(gray, h=strength)


def deskew(gray):
    """Detects the dominant text angle and rotates the image to straighten it.
    Falls back to the original image if no clear angle is found (e.g. a
    photo with very little text)."""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) < 50:
        return gray, 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    angle = -angle

    if abs(angle) < 0.5:
        return gray, 0.0

    (h, w) = gray.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def binarize(gray):
    """Off by default — testing showed this hurts accuracy on rotated or
    blurred text more than it helps, since it destroys the antialiasing
    detail Tesseract's own internal thresholding relies on. Kept as an
    option for document types where it does help, like scans with very
    uneven lighting."""
    _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return result


def preprocess(image_path, do_denoise=True, do_deskew=True, do_binarize=False):
    """Runs the pipeline on an image file and returns the processed image
    as a numpy array, ready for OCR. Also returns a dict of what happened,
    for debugging."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    info = {}
    gray = to_grayscale(img)

    if do_denoise:
        gray = denoise(gray)

    if do_deskew:
        gray, angle = deskew(gray)
        info["deskew_angle"] = round(angle, 2)

    if do_binarize:
        gray = binarize(gray)

    return gray, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="preprocessed_debug.png")
    parser.add_argument("--binarize", action="store_true", help="Force Otsu binarization on")
    args = parser.parse_args()

    result, info = preprocess(args.image, do_binarize=args.binarize)
    cv2.imwrite(args.out, result)
    print(f"Saved preprocessed image to {args.out}")
    print(f"Info: {info}")


if __name__ == "__main__":
    main()