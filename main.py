
import argparse
import json

from recognize import recognize_text
from extract_fields import extract_fields


def run(image_path, mode="text", binarize=False):
    result = recognize_text(image_path, do_binarize=binarize)
    output = {
        "text": result["text"],
        "confidence": result["confidence"],
        "preprocessing": result["preprocessing"],
    }

    if mode == "id":
        output["fields"] = extract_fields(result["text"])

    return output


def main():
    parser = argparse.ArgumentParser(description="Extract text from any image.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument(
        "--mode", choices=["text", "id"], default="text",
        help="'text' (default): just extract raw text. "
             "'id': also try to pull structured fields (name, DOB, etc.)",
    )
    parser.add_argument("--binarize", action="store_true", help="Force Otsu binarization on")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    output = run(args.image, mode=args.mode, binarize=args.binarize)

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print(f"Confidence: {output['confidence']}%")
    if output["preprocessing"].get("deskew_angle"):
        print(f"Corrected skew: {output['preprocessing']['deskew_angle']}°")
    print()

    if args.mode == "id":
        if output["fields"]:
            for f in output["fields"]:
                print(f"{f['label']}: {f['value']}")
        else:
            print("No labeled fields found — this may not be a structured document.")
            print()
            print("Raw text:")
            print(output["text"])
    else:
        print(output["text"])


if __name__ == "__main__":
    main()