import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from google import genai


def parse_args():
    parser = argparse.ArgumentParser(description="Extract generated images from a batch JSONL result file.")
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/synthetic_batch"))
    parser.add_argument("--result-jsonl", type=Path, default=None)
    parser.add_argument("--images-dir", type=Path, default=None)
    return parser.parse_args()


def decode_inline_image_data(data):
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return base64.b64decode(data)
    raise TypeError(f"Unsupported inline image data type: {type(data)}")


def extract_image_from_part(part, client):
    if not isinstance(part, dict):
        return None

    inline_data = part.get("inlineData") or part.get("inline_data")
    if inline_data:
        mime_type = inline_data.get("mimeType") or inline_data.get("mime_type", "")
        if mime_type.startswith("image/") and "data" in inline_data:
            raw = decode_inline_image_data(inline_data["data"])
            return Image.open(BytesIO(raw))

    file_data = part.get("fileData") or part.get("file_data")
    if file_data:
        mime_type = file_data.get("mimeType") or file_data.get("mime_type", "")
        file_uri = file_data.get("fileUri") or file_data.get("file_uri")
        if mime_type.startswith("image/") and file_uri:
            raw = client.files.download(file=file_uri)
            return Image.open(BytesIO(raw))

    return None


def extract_image_from_item(item, client):
    for top_key in ["response", "result", "modelResponse"]:
        obj = item.get(top_key)
        if not isinstance(obj, dict):
            continue

        for cand in obj.get("candidates", []):
            content = cand.get("content", {})
            for part in content.get("parts", []):
                img = extract_image_from_part(part, client)
                if img is not None:
                    return img

        for part in obj.get("parts", []):
            img = extract_image_from_part(part, client)
            if img is not None:
                return img

        content = obj.get("content", {})
        for part in content.get("parts", []):
            img = extract_image_from_part(part, client)
            if img is not None:
                return img

    return None


def main():
    args = parse_args()
    result_jsonl = args.result_jsonl or args.batch_dir / "batch_results.jsonl"
    images_dir = args.images_dir or args.batch_dir / "generated_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if not result_jsonl.exists():
        raise FileNotFoundError(f"Missing result file: {result_jsonl}")

    client = genai.Client()
    saved = 0
    failed = 0

    with open(result_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                key = item.get("key", f"line_{line_num}")
                out_path = images_dir / f"{key}.png"

                img = extract_image_from_item(item, client)
                if img is None:
                    print(f"No image found for {key}")
                    failed += 1
                    continue

                img.load()
                img.save(out_path)
                print(f"Saved: {out_path} | size={img.size}")
                saved += 1

            except Exception as exc:
                print(f"Error on line {line_num}: {exc}")
                failed += 1

    print(f"\nSaved: {saved}")
    print(f"Failed: {failed}")
    print(f"Output dir: {images_dir}")


if __name__ == "__main__":
    main()
