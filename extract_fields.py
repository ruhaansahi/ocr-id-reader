

import argparse
import re

from recognize import recognize_text


LABEL_TEXT_RE = re.compile(r"^[A-Za-z&][A-Za-z &]{1,38}$")


def _is_all_caps_label(line):
    
    line = line.strip()
    return bool(LABEL_TEXT_RE.match(line)) and line.isupper()


def _looks_like_label(line):
    
    line = line.strip()
    if not LABEL_TEXT_RE.match(line):
        return False
    words = line.split()
    if not words:
        return False
    all_caps = line.isupper()
    title_case = all(w[0].isupper() for w in words if w)
    return all_caps or title_case


def extract_fields(text):
    
    lines = [ln.strip() for ln in text.splitlines()]
    fields = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line:
            i += 1
            continue

        if ":" in line:
            label, _, value = line.partition(":")
            label, value = label.strip(), value.strip()
            if label and value and len(label) <= 40 and not label[0].isdigit():
                fields.append({"label": label, "value": value})
                i += 1
                continue

       
        match = re.search(r"[\d$]", line)
        if match and match.start() > 0 and not line[0].isdigit():
            label_part = line[: match.start()].strip()
            value_part = line[match.start():].strip()
            if label_part and value_part and _looks_like_label(label_part):
                fields.append({"label": label_part, "value": value_part})
                i += 1
                continue


        if _looks_like_label(line) and i + 1 < len(lines):
            next_line = lines[i + 1]
            if next_line and ":" not in next_line and not _is_all_caps_label(next_line):
                fields.append({"label": line, "value": next_line})
                i += 2
                continue

        i += 1

    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    result = recognize_text(args.image)
    fields = extract_fields(result["text"])

    if not fields:
        print("No labeled fields found — this may not be a structured document.")
        return

    for f in fields:
        print(f"{f['label']}: {f['value']}")


if __name__ == "__main__":
    main()