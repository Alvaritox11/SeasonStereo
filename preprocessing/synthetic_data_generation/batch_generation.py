import argparse
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from google import genai
from google.genai import types

from utils import (
    ASPECT_RATIO,
    IMAGE_SIZE,
    PROMPTS,
    get_aoi_dirs_from_txt,
    get_images_from_aoi_dirs,
    resolve_track_roots,
    save_jsonl,
    tif_to_temp_png,
)


DEFAULT_MODEL = "gemini-3-pro-image-preview"

IMAGE_INPUT_COST = 0.0006
IMAGE_OUTPUT_COST_1K_2K = 0.067
TEXT_INPUT_COST_PER_MTOK = 1.00


def parse_args():
    parser = argparse.ArgumentParser(description="Submit seasonal image generation as a batch job.")
    parser.add_argument("--aoi-list", type=Path, default=Path("selected_aois.txt"))
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/synthetic_batch"))
    parser.add_argument("--track3-root", type=Path, default=Path("data/Train-Track3-cropped"))
    parser.add_argument("--track-root", type=Path, action="append", default=None,
                        help="Explicit Track3-RGB root. Can be passed multiple times.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    return parser.parse_args()


def upload_input_files(client, temp_png_paths):
    uploaded = {}
    for png_path in temp_png_paths:
        print(f"Uploading {png_path}")
        file_obj = client.files.upload(
            file=str(png_path),
            config=types.UploadFileConfig(
                display_name=png_path.stem,
                mime_type="image/png",
            ),
        )
        uploaded[png_path] = file_obj
        print(f"  name={file_obj.name}")
        print(f"  uri={file_obj.uri}")
    return uploaded


def build_batch_records(uploaded_files, jsonl_path: Path):
    records = []
    for png_path, file_obj in uploaded_files.items():
        for season, prompt_text in PROMPTS.items():
            key = f"{png_path.stem}_{season}"
            records.append(
                {
                    "key": key,
                    "request": {
                        "contents": [
                            {
                                "parts": [
                                    {"text": prompt_text},
                                    {
                                        "file_data": {
                                            "mime_type": file_obj.mime_type or "image/png",
                                            "file_uri": file_obj.uri,
                                        }
                                    },
                                ]
                            }
                        ],
                        "generation_config": {
                            "responseModalities": ["IMAGE"],
                            "temperature": 0.2,
                            "imageConfig": {
                                "aspectRatio": ASPECT_RATIO,
                                "imageSize": IMAGE_SIZE,
                            },
                        },
                    },
                }
            )

    save_jsonl(records, jsonl_path)
    print(f"\nSaved JSONL: {jsonl_path}")
    print(f"Total requests: {len(records)}")
    return records


def estimate_cost(records):
    total_requests = len(records)
    image_input_cost = total_requests * IMAGE_INPUT_COST
    image_output_cost = total_requests * IMAGE_OUTPUT_COST_1K_2K
    total_prompt_chars = sum(len(rec["request"]["contents"][0]["parts"][0]["text"]) for rec in records)
    approx_text_tokens = total_prompt_chars / 4.0
    approx_text_cost = (approx_text_tokens / 1_000_000) * TEXT_INPUT_COST_PER_MTOK
    total_estimated = image_input_cost + image_output_cost + approx_text_cost

    print("\n=== Estimated cost ===")
    print(f"Requests: {total_requests}")
    print(f"Image input cost:  ${image_input_cost:.6f}")
    print(f"Image output cost: ${image_output_cost:.6f}")
    print(f"Approx text cost:  ${approx_text_cost:.6f}")
    print(f"Estimated total:   ${total_estimated:.6f}")


def submit_batch(client, jsonl_path: Path, model_name: str):
    uploaded_jsonl = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(
            display_name=jsonl_path.stem,
            mime_type="jsonl",
        ),
    )
    print(f"\nUploaded JSONL file: {uploaded_jsonl.name}")

    batch_job = client.batches.create(
        model=model_name,
        src=uploaded_jsonl.name,
        config={"display_name": "seasonal-satellite-generation"},
    )
    print(f"Created batch job: {batch_job.name}")
    return batch_job.name


def poll_batch(client, job_name: str, poll_interval: float):
    done_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        print(f"Batch state: {state}")
        if state in done_states:
            return batch_job
        time.sleep(poll_interval)


def download_results(client, batch_job, batch_dir: Path):
    if batch_job.state.name != "JOB_STATE_SUCCEEDED":
        print(f"Batch did not succeed: {batch_job.state.name}")
        if getattr(batch_job, "error", None):
            print(batch_job.error)
        return

    result_file_name = batch_job.dest.file_name
    raw = client.files.download(file=result_file_name)
    result_path = batch_dir / "batch_results.jsonl"
    result_path.write_bytes(raw)
    print(f"Saved results to: {result_path}")


def main():
    args = parse_args()
    args.batch_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.batch_dir / "aoi_requests.jsonl"

    track_roots = resolve_track_roots(args.track3_root, args.track_root)
    selected_aoi_dirs = get_aoi_dirs_from_txt(track_roots, args.aoi_list)

    print("Selected AOI folders:")
    for aoi_dir in selected_aoi_dirs:
        print(f"  {aoi_dir}")

    aoi_tifs = get_images_from_aoi_dirs(selected_aoi_dirs)
    print(f"\nTotal TIFF images found across selected AOIs: {len(aoi_tifs)}")
    if not aoi_tifs:
        raise ValueError("No TIFF images found for the selected AOIs.")

    client = genai.Client()
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        temp_png_paths = [tif_to_temp_png(tif_path, temp_dir) for tif_path in aoi_tifs]
        uploaded_files = upload_input_files(client, temp_png_paths)
        records = build_batch_records(uploaded_files, jsonl_path)
        estimate_cost(records)

        job_name = submit_batch(client, jsonl_path, args.model)
        batch_job = poll_batch(client, job_name, args.poll_interval)
        download_results(client, batch_job, args.batch_dir)


if __name__ == "__main__":
    main()
