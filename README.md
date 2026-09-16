# OCR Text Extraction Pipeline

A general-purpose OCR tool: give it any image and it extracts the text.
It also has an optional mode for pulling structured fields (name, date,
ID number, etc.) out of documents that have a label/value layout, like
ID cards or receipts.

## What it can do

- **Extract text from any image** — photos, screenshots, scanned documents, receipts. Not limited to any one document type.
- **Automatically straighten tilted images** — detects rotation and corrects it before running OCR, so a photo taken at a slight angle still reads correctly.
- **Pull out structured fields** — for documents with a label/value shape ("NAME" / "Ruhaan Sahi", or "TOTAL AMOUNT $117.00"), it returns them as clean key-value pairs instead of a wall of text. Works generically — it isn't hardcoded to ID cards specifically, and correctly returns nothing when an image has no such structure.
- **Report a confidence score** — every result comes with a 0-100 confidence number, computed from Tesseract's own per-word confidence values.
- **Generate synthetic test data** — a script produces fake ID-card images with known correct answers, at three difficulty levels (clean, slightly rotated, rotated + blurred), so the pipeline can be tested without ever touching real personal data.
- **Measure its own accuracy** — an evaluation script runs the full pipeline against the synthetic dataset and reports character error rate, word error rate, and per-field accuracy, broken down by difficulty tier.
- **Command-line interface** — one entry point (`main.py`) ties everything together, with plain-text or JSON output.

## Libraries used

**Pillow (PIL)** — a general Python library for creating and editing images (drawing shapes, rendering text, applying filters). Used in `generate_synthetic_data.py` to draw the fake ID cards from scratch: the card background, header bar, label/value text, and the rotation/blur applied for the harder difficulty tiers.

**OpenCV (`cv2`)** — a computer vision library used for image processing operations. Used in `preprocess.py` for the whole cleanup pipeline: converting to grayscale, denoising (removing camera/scan grain), detecting and correcting rotation (deskewing, via Otsu thresholding and `minAreaRect`), and an optional binarization step.

**NumPy** — the standard numerical array library for Python. Used underneath OpenCV's operations, particularly for handling pixel coordinate arrays during deskew angle detection.

**pytesseract** — a Python wrapper around the Tesseract OCR engine. This is the library that actually performs character recognition, in `recognize.py`: `image_to_string` returns the extracted text, `image_to_data` returns per-word confidence scores.

## Tools used

**Tesseract OCR** — the underlying open-source OCR engine that does the actual recognition. It's a separate system binary (installed via Homebrew on Mac), not a Python package — pytesseract just calls out to it.

**Bundled fonts (DejaVu Sans)** — shipped in `assets/fonts/` and used by the synthetic data generator instead of relying on whatever font happens to be installed on a given machine. Testing showed that different default fonts (Arial on Mac vs. DejaVu on Linux) produced meaningfully different OCR accuracy on what was supposed to be the same benchmark — bundling the font makes results reproducible across machines.

**Git** — version control.

## Limitations

- **Fails on stylized signage and scene text.** Tested against a real billboard photo with large 3D-perspective lettering over a busy background — Tesseract returned garbled fragments, not real text. It's built for flat printed or scanned text, not text photographed at an angle in a natural scene. This would need a dedicated scene-text detection model to handle properly.
- **Blur destroys alphanumeric field accuracy badly.** On the synthetic "hard" difficulty tier (rotation + blur), ID number extraction accuracy drops to ~2%, even though names and dates hold up much better on the same images. A single misread character breaks an exact-match field like an ID number, while it's more forgivable in a name or date. Upscaling and sharpening were tested as fixes and only produced modest gains (~5-7 percentage points).
- **No text localization stage.** The tool assumes the image is already roughly text-bearing; it doesn't find and crop text regions out of a larger, busier photo first.
- **Field extraction is rule-based, not learned.** `extract_fields.py` uses regex and formatting heuristics (ALL CAPS labels, colon separators) to find label/value pairs. It works well on documents shaped like the ones tested, but isn't a trained model and won't generalize to arbitrarily formatted documents.
- **No handwriting support.** Tesseract is built for printed text; handwriting accuracy is poor.
- **Limited file format support.** OpenCV can't read HEIC or AVIF images directly — they need to be converted to PNG/JPG first (e.g. with macOS's `sips` command).
- **Binarization is off by default.** An Otsu binarization step exists in `preprocess.py` but isn't applied by default, since testing showed it reduces accuracy on rotated or blurred text more than it helps — it destroys antialiasing detail Tesseract's own internal thresholding relies on.

## Planned future work

- **Train a custom Keras/TensorFlow model** (CNN + RNN with CTC loss) for character recognition, aimed specifically at improving accuracy on blurred images, where Tesseract currently struggles most.
- **Evaluate `keras-ocr`** as a pretrained alternative for the scene-text/signage case that Tesseract currently fails on.
- **Add an optional upscaling step** to `preprocess.py`, since testing showed a small but real accuracy improvement on degraded images.
- **Character-segmentation pipeline** for reading full handwritten words (printed/block letters first), built on top of a trained single-character classifier.
