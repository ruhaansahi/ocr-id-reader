
import argparse

import pytesseract
from pytesseract import Output

from preprocess import preprocess


def recognize_text(image_path, do_denoise=True, do_deskew=True, do_binarize=False):
    
    img, info = preprocess(
        image_path, do_denoise=do_denoise, do_deskew=do_deskew, do_binarize=do_binarize
    )

    text = pytesseract.image_to_string(img).strip()

    data = pytesseract.image_to_data(img, output_type=Output.DICT)
    confidences = [c for c in data["conf"] if c != -1]
    mean_confidence = round(sum(confidences) / len(confidences), 1) if confidences else 0.0

    return {
        "text": text,
        "confidence": mean_confidence,
        "preprocessing": info,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--binarize", action="store_true", help="Force Otsu binarization on")
    args = parser.parse_args()

    result = recognize_text(args.image, do_binarize=args.binarize)

    print(f"--- Preprocessing: {result['preprocessing']} ---")
    print(f"--- Confidence: {result['confidence']}% ---")
    print(result["text"])


if __name__ == "__main__":
    main()