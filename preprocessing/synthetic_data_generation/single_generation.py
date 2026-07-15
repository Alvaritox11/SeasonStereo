import argparse
import time
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from google import genai
from google.genai import types

from utils import (
    ASPECT_RATIO,
    IMAGE_SIZE,
    PROMPTS,
    choose_first_n_aoi_dirs,
    get_images_from_aoi_dirs,
    resolve_track_roots,
    tif_to_temp_png,
)


DEFAULT_MODEL = "gemini-3-pro-image-preview"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preview seasonal generations from one image or a few AOI folders."
    )
    parser.add_argument("--image-path", type=Path, default=None, help="Path to a specific AOI TIFF image.")
    parser.add_argument("--seasons", nargs="+", default=["WINTER", "SUMMER"], help="Seasons to generate.")
    parser.add_argument("--num-aois", type=int, default=2, help="Number of AOI folders to preview without --image-path.")
    parser.add_argument("--track3-root", type=Path, default=Path("data/Train-Track3-cropped"))
    parser.add_argument("--track-root", type=Path, action="append", default=None,
                        help="Explicit Track3-RGB root. Can be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_preview"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between requests.")
    return parser.parse_args()


def validate_seasons(seasons):
    invalid = [season for season in seasons if season not in PROMPTS]
    if invalid:
        raise ValueError(f"Invalid seasons: {invalid}. Valid options are: {list(PROMPTS.keys())}")


def resolve_input_files(args, track_roots):
    if args.image_path is not None:
        tif_path = args.image_path
        if not tif_path.exists():
            raise FileNotFoundError(f"Image not found: {tif_path}")
        if tif_path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError(f"Expected a TIFF image, got: {tif_path}")
        return [tif_path]

    selected_aoi_dirs = choose_first_n_aoi_dirs(track_roots, n=args.num_aois)
    print("Selected AOI folders:")
    for aoi_dir in selected_aoi_dirs:
        print(f"  {aoi_dir}")

    input_files = get_images_from_aoi_dirs(selected_aoi_dirs)
    if not input_files:
        raise ValueError("No TIFF images found in the selected AOI folders.")
    return input_files


def generate_for_image(client, model_name: str, tif_path: Path, seasons, sleep_seconds: float,
                       temp_dir: Path, output_dir: Path):
    png_path = tif_to_temp_png(tif_path, temp_dir)

    with Image.open(png_path) as input_image:
        input_image = input_image.convert("RGB")
        input_width, input_height = input_image.size

        if (input_width, input_height) != (1024, 1024):
            raise ValueError(
                f"Expected converted preview input to be 1024x1024, got {input_width}x{input_height}"
            )

        for season in seasons:
            prompt_text = PROMPTS[season]
            print(f"\n--- Generating {season} for {tif_path} ---")

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[prompt_text, input_image],
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                        temperature=0.2,
                        image_config=types.ImageConfig(
                            aspect_ratio=ASPECT_RATIO,
                            image_size=IMAGE_SIZE,
                        ),
                    ),
                )

                image_saved = False
                for part in response.parts:
                    if getattr(part, "inline_data", None):
                        generated_img = Image.open(BytesIO(part.inline_data.data))
                        out_path = output_dir / f"{tif_path.stem}_{season}.png"

                        print(f"Input size:  {input_width}x{input_height}")
                        print(f"Output size: {generated_img.size[0]}x{generated_img.size[1]}")

                        if generated_img.size != (input_width, input_height):
                            print("Warning: resizing output back to input size.")
                            generated_img = generated_img.resize((input_width, input_height), Image.LANCZOS)

                        generated_img.save(out_path)
                        print(f"Saved: {out_path}")
                        image_saved = True

                if not image_saved:
                    print(f"No image returned for {season}")

            except Exception as exc:
                print(f"Error generating {season} for {tif_path.name}: {exc}")

            time.sleep(sleep_seconds)


def main():
    args = parse_args()
    validate_seasons(args.seasons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    track_roots = resolve_track_roots(args.track3_root, args.track_root)
    input_files = resolve_input_files(args, track_roots)

    print("\nSelected input files:")
    for path in input_files:
        print(f"  {path}")

    client = genai.Client()
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for tif_path in input_files:
            generate_for_image(client, args.model, tif_path, args.seasons, args.sleep, temp_dir, args.output_dir)


if __name__ == "__main__":
    main()
