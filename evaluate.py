
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from recognize import recognize_text
from extract_fields import extract_fields


def levenshtein(a, b):
 
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,      # deletion
                current[j - 1] + 1,   # insertion
                previous[j - 1] + cost,  # substitution
            ))
        previous = current
    return previous[-1]


def cer(reference, hypothesis):
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


CANONICAL_FIELDS = {
    "name": lambda l: "name" in l,
    "dob": lambda l: "birth" in l or "dob" in l,
    "id_number": lambda l: "id" in l and ("num" in l or "no" in l),
    "expiry": lambda l: "expir" in l,
}


def match_canonical_field(label):
    l = label.lower()
    for canonical, test in CANONICAL_FIELDS.items():
        if test(l):
            return canonical
    return None


def ground_truth_text(row):
    
    return (
        f"IDENTIFICATION CARD\nNAME\n{row['name']}\nDATE OF BIRTH\n{row['dob']}\n"
        f"ID NUMBER\n{row['id_number']}\nEXPIRES\n{row['expiry']}"
    )


def evaluate_image(image_path, row):
    result = recognize_text(image_path)
    ocr_text = result["text"]

    gt_text = ground_truth_text(row)
    char_error = cer(gt_text, ocr_text)
    word_error = wer(gt_text, ocr_text)

    fields = extract_fields(ocr_text)
    extracted = {}
    for f in fields:
        canonical = match_canonical_field(f["label"])
        if canonical and canonical not in extracted:
            extracted[canonical] = f["value"]

    field_correct = {
        "name": extracted.get("name") == row["name"],
        "dob": extracted.get("dob") == row["dob"],
        "id_number": extracted.get("id_number") == row["id_number"],
        "expiry": extracted.get("expiry") == row["expiry"],
    }

    return {
        "cer": char_error,
        "wer": word_error,
        "confidence": result["confidence"],
        "field_correct": field_correct,
        "fields_found": len(fields),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/synthetic/labels.csv")
    parser.add_argument("--images", default="data/synthetic/images")
    args = parser.parse_args()

    images_dir = Path(args.images)
    rows = list(csv.DictReader(open(args.labels)))

    by_difficulty = defaultdict(list)

    for row in rows:
        image_path = images_dir / row["filename"]
        result = evaluate_image(image_path, row)
        by_difficulty[row["difficulty"]].append(result)
        by_difficulty["overall"].append(result)

    print(f"{'Tier':<10} {'N':<5} {'Mean CER':<10} {'Mean WER':<10} {'Mean Conf':<10} "
          f"{'Name':<7} {'DOB':<7} {'ID#':<7} {'Expiry':<7}")

    for tier in ["clean", "medium", "hard", "overall"]:
        results = by_difficulty[tier]
        if not results:
            continue
        n = len(results)
        mean_cer = sum(r["cer"] for r in results) / n
        mean_wer = sum(r["wer"] for r in results) / n
        mean_conf = sum(r["confidence"] for r in results) / n

        field_acc = {}
        for field in ["name", "dob", "id_number", "expiry"]:
            correct = sum(1 for r in results if r["field_correct"][field])
            field_acc[field] = correct / n

        print(f"{tier:<10} {n:<5} {mean_cer:<10.3f} {mean_wer:<10.3f} {mean_conf:<10.1f} "
              f"{field_acc['name']:<7.1%} {field_acc['dob']:<7.1%} "
              f"{field_acc['id_number']:<7.1%} {field_acc['expiry']:<7.1%}")


if __name__ == "__main__":
    main()