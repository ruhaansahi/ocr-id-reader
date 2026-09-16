"""
Generates fake ID-card images with randomized names, ID numbers, and
dates, plus a ground-truth label file. This is the test data for the
rest of the pipeline — real IDs are personal data, so everything here
is invented.

Usage:
    python3 generate_synthetic_data.py --count 200
"""

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

FIRST_NAMES = [
    "James", "Maria", "Ahmed", "Yuki", "Fatima", "Liam", "Sofia", "Noah",
    "Amara", "Lucas", "Priya", "Omar", "Chen", "Elena", "Kwame", "Mei",
    "Diego", "Aisha", "Ivan", "Zara",
]
LAST_NAMES = [
    "Smith", "Khan", "Garcia", "Tanaka", "Hassan", "Müller", "Silva",
    "Kim", "Okafor", "Rossi", "Patel", "Nguyen", "Petrov", "Costa",
    "Andersson", "Brown", "Hernandez", "Ali", "Novak", "Reed",
]

CARD_SIZE = (640, 400)
BACKGROUND_COLOR = (223, 232, 240)
HEADER_COLOR = (36, 62, 92)
TEXT_COLOR = (20, 20, 20)


def find_fonts():
    """Uses the fonts bundled in assets/fonts/ so every machine renders
    identical images — relying on whatever font happens to be installed
    per-OS (Arial on Mac, DejaVu on Linux, etc.) produced visibly
    different text rendering and, in testing, noticeably different OCR
    accuracy on the same "image". Falls back to system fonts, then
    PIL's built-in bitmap font, only if the bundled files are missing."""
    script_dir = Path(__file__).resolve().parent
    bundled_bold = script_dir / "assets" / "fonts" / "DejaVuSans-Bold.ttf"
    bundled_regular = script_dir / "assets" / "fonts" / "DejaVuSans.ttf"

    system_candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]

    if bundled_bold.exists() and bundled_regular.exists():
        bold_path, regular_path = str(bundled_bold), str(bundled_regular)
    else:
        bold_path = next((p for p in system_candidates if "Bold" in p or "bd" in p), None)
        regular_path = next((p for p in system_candidates if "Bold" not in p and "bd" not in p), None)

    def load(path, size):
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    return {
        "title": load(bold_path, 22),
        "label": load(bold_path, 14),
        "value": load(regular_path, 18),
    }


def random_person():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    dob = date.today() - timedelta(days=random.randint(18 * 365, 70 * 365))
    expiry = date.today() + timedelta(days=random.randint(1 * 365, 10 * 365))
    id_number = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2)) + \
        "".join(random.choices("0123456789", k=7))
    return {
        "name": f"{first} {last}",
        "dob": dob.strftime("%d %b %Y"),
        "expiry": expiry.strftime("%d %b %Y"),
        "id_number": id_number,
    }


def render_card(person, fonts):
    img = Image.new("RGB", CARD_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, CARD_SIZE[0], 60], fill=HEADER_COLOR)
    draw.text((20, 18), "IDENTIFICATION CARD", font=fonts["title"], fill="white")

    # photo placeholder
    draw.rectangle([20, 90, 180, 300], outline=(120, 120, 120), width=2)
    draw.line([20, 90, 180, 300], fill=(180, 180, 180), width=1)
    draw.line([180, 90, 20, 300], fill=(180, 180, 180), width=1)

    rows = [
        ("NAME", person["name"]),
        ("DATE OF BIRTH", person["dob"]),
        ("ID NUMBER", person["id_number"]),
        ("EXPIRES", person["expiry"]),
    ]
    y = 100
    for label, value in rows:
        draw.text((210, y), label, font=fonts["label"], fill=(90, 90, 90))
        draw.text((210, y + 18), value, font=fonts["value"], fill=TEXT_COLOR)
        y += 60

    return img


def apply_noise(img, difficulty):
    """difficulty: 'clean', 'medium', or 'hard' — how much distortion to add."""
    if difficulty == "clean":
        return img

    img = img.rotate(random.uniform(-4, 4) if difficulty == "medium" else random.uniform(-9, 9),
                      expand=False, fillcolor=BACKGROUND_COLOR)

    if difficulty == "hard":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=150, help="Number of images to generate")
    parser.add_argument("--out-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fonts = find_fonts()
    difficulties = ["clean", "medium", "hard"]

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "difficulty", "name", "dob", "id_number", "expiry"])

        for i in range(args.count):
            person = random_person()
            difficulty = difficulties[i % len(difficulties)]
            img = render_card(person, fonts)
            img = apply_noise(img, difficulty)

            filename = f"id_{i:04d}.png"
            img.save(images_dir / filename)
            writer.writerow([filename, difficulty, person["name"], person["dob"],
                              person["id_number"], person["expiry"]])

    print(f"Generated {args.count} synthetic ID images in {images_dir}")
    print(f"Ground-truth labels written to {labels_path}")


if __name__ == "__main__":
    main()