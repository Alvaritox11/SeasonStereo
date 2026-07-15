#!/usr/bin/env python3
"""Generate rectified training data from downloaded release inputs."""

from __future__ import annotations

import argparse
import glob
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import iio
import numpy as np
from PIL import Image
from skimage import transform


FULL_SHAPE = (2048, 2048)
CROP_SIZE = 704

REAL_EXTS = (".tif", ".tiff")
SYNTHETIC_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")

REGION_DIRS = {
    "JAX": "Track3-RGB-1",
    "OMA": "Track3-RGB-2",
}


@dataclass(frozen=True)
class Roots:
    track3_root: Path
    synthetic_root: Path
    water_masks_dir: Path
    tree_masks_dir: Path
    building_masks_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    name: str
    reference_dir: Path
    output_dir: Path
    include_synthetic: bool


@dataclass(frozen=True)
class Candidate:
    tag: str
    path: Path
    is_real: bool


class SplitLogger:
    def __init__(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.missing = output_dir / "missing_files.txt"
        self.errors = output_dir / "errors.txt"
        self.summary = output_dir / "summary.txt"
        for path in (self.missing, self.errors, self.summary):
            path.write_text("")

    def log_missing(self, message: str) -> None:
        with self.missing.open("a") as f:
            f.write(message.rstrip() + "\n")

    def log_error(self, message: str) -> None:
        with self.errors.open("a") as f:
            f.write(message.rstrip() + "\n")

    def log_summary(self, message: str) -> None:
        with self.summary.open("a") as f:
            f.write(message.rstrip() + "\n")


def warp_with_h(image: np.ndarray, homography: np.ndarray, output_shape: Tuple[int, int], order: int) -> np.ndarray:
    return transform.warp(
        image,
        transform.ProjectiveTransform(np.asarray(homography, dtype=np.float64)).inverse,
        output_shape=output_shape,
        preserve_range=True,
        mode="constant",
        cval=0,
        order=order,
    )


def bounding_box_2d(points: np.ndarray) -> Tuple[float, float, float, float]:
    points = np.asarray(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return float(mins[0]), float(mins[1]), float(maxs[0] - mins[0]), float(maxs[1] - mins[1])


def translation(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def h_crop_to_dst_global(h_full: np.ndarray, full_shape: Tuple[int, int], crop_shape: Tuple[int, int]) -> np.ndarray:
    full_h, full_w = full_shape
    crop_h, crop_w = crop_shape
    cx = (full_w - crop_w) / 2.0
    cy = (full_h - crop_h) / 2.0
    return h_full @ translation(cx, cy)


def compute_output_shape(image_shape: Sequence[int], h_left: np.ndarray) -> Tuple[int, int]:
    image_h, image_w = image_shape[:2]
    crop_size = CROP_SIZE if image_h >= CROP_SIZE and image_w >= CROP_SIZE else 512
    x = image_w // 2 - crop_size // 2
    y = image_h // 2 - crop_size // 2
    w = h = crop_size

    roi = np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    roi_after = cv2.perspectiveTransform(roi, h_left)
    x_f, y_f, w_f, h_f = bounding_box_2d(roi_after.reshape(-1, 2))
    _, _, w_bb, h_bb = np.round([x_f, y_f, w_f, h_f]).astype(int)
    return int(h_bb), int(w_bb)


def adapt_homography_for_shape(image_shape: Sequence[int], h_full: np.ndarray) -> np.ndarray:
    if tuple(image_shape[:2]) != FULL_SHAPE:
        return h_crop_to_dst_global(h_full, FULL_SHAPE, tuple(image_shape[:2]))
    return h_full


def parse_pair_stem(pair_stem: str) -> Tuple[str, str]:
    if "-" not in pair_stem:
        raise ValueError(f"Unexpected pair stem without '-': {pair_stem}")
    return tuple(pair_stem.split("-", 1))  # type: ignore[return-value]


def folder_from_stem(stem: str) -> Tuple[str, str]:
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected image stem: {stem}")
    return parts[0], f"{parts[0]}_{parts[1]}"


def region_root(base_root: Path, region: str) -> Path:
    try:
        return base_root / REGION_DIRS[region]
    except KeyError as exc:
        raise ValueError(f"Unsupported region: {region}") from exc


def resolve_source_image(stem: str, track3_root: Path, exts: Sequence[str]) -> Path:
    region, folder = folder_from_stem(stem)
    root = region_root(track3_root, region)
    for ext in exts:
        path = root / folder / f"{stem}{ext}"
        if path.exists():
            return path
    return root / folder / f"{stem}{exts[0]}"


def find_synthetic_variants(stem: str, synthetic_root: Path) -> Dict[str, Path]:
    region, folder = folder_from_stem(stem)
    folder_path = region_root(synthetic_root, region) / folder
    if not folder_path.is_dir():
        return {}

    variants: Dict[str, Path] = {}
    prefix = f"{stem}_"
    for ext in SYNTHETIC_EXTS:
        for match in glob.glob(str(folder_path / f"{prefix}*{ext}")):
            path = Path(match)
            suffix = path.stem[len(prefix):]
            if suffix:
                variants[suffix] = path
    return dict(sorted(variants.items()))


def image_candidates(stem: str, roots: Roots) -> List[Candidate]:
    candidates: List[Candidate] = []

    real_path = resolve_source_image(stem, roots.track3_root, REAL_EXTS)
    if real_path.exists():
        candidates.append(Candidate("real", real_path, True))

    for suffix, path in find_synthetic_variants(stem, roots.synthetic_root).items():
        candidates.append(Candidate(suffix, path, False))

    return candidates


def synthetic_jobs(left_stem: str, right_stem: str, roots: Roots) -> List[Dict[str, object]]:
    jobs: List[Dict[str, object]] = []
    for left in image_candidates(left_stem, roots):
        for right in image_candidates(right_stem, roots):
            if left.is_real and right.is_real:
                continue
            left_name = left_stem if left.is_real else f"{left_stem}__{left.tag}"
            right_name = right_stem if right.is_real else f"{right_stem}__{right.tag}"
            jobs.append(
                {
                    "out_stem": f"{left_name}-{right_name}",
                    "left_path": left.path,
                    "right_path": right.path,
                }
            )
    return jobs


def read_gray(path: Path) -> np.ndarray:
    arr = iio.read(str(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.asarray(arr)


def load_keep_mask(path: Path) -> np.ndarray:
    arr = read_gray(path)
    if arr.max() <= 1:
        return (arr > 0).astype(np.uint8)
    return (arr > 127).astype(np.uint8)


def load_building_positive_mask(path: Path) -> np.ndarray:
    arr = read_gray(path)
    if arr.max() <= 1:
        return (arr == 0).astype(np.uint8)
    return (arr <= 127).astype(np.uint8)


def resolve_mask_path(stem: str, mask_dir: Path) -> Path:
    return mask_dir / f"{stem}_mask.png"


def save_png_uint8(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 3 and arr.shape[2] == 3:
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)
    elif arr.ndim == 2:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)
    else:
        raise ValueError(f"Unsupported PNG shape: {arr.shape}")


def ensure_output_dirs(output_dir: Path, save_preview_png: bool) -> None:
    required = [
        "L", "R",
        "masks/L", "masks/R",
        "water_masks/L", "water_masks/R",
        "tree_masks/L", "tree_masks/R",
        "building_masks/L", "building_masks/R",
        "homography",
    ]
    if save_preview_png:
        required.extend(
            [
                "png/L", "png/R",
                "png/masks/L", "png/masks/R",
                "png/water_masks/L", "png/water_masks/R",
                "png/tree_masks/L", "png/tree_masks/R",
                "png/building_masks/L", "png/building_masks/R",
            ]
        )
    for relative in required:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def rectify_optional_mask(
    mask_path: Path,
    image_shape: Sequence[int],
    homography: np.ndarray,
    out_shape: Tuple[int, int],
    loader,
    default_value: int,
) -> np.ndarray:
    if mask_path.exists():
        mask = loader(mask_path)
        if mask.shape[:2] != tuple(image_shape[:2]):
            raise ValueError(
                f"Mask/source shape mismatch: mask={mask.shape[:2]} src={tuple(image_shape[:2])} | {mask_path}"
            )
    else:
        mask = np.full(tuple(image_shape[:2]), default_value, dtype=np.uint8)

    warped = warp_with_h(mask.astype(np.float32), homography, out_shape, order=0)
    return (warped > 0.5).astype(np.uint8)


def compute_shared_geometry(
    src_left: Path,
    src_right: Path,
    h_left_full: np.ndarray,
    h_right_full: np.ndarray,
    left_stem: str,
    right_stem: str,
    roots: Roots,
    logger: SplitLogger,
) -> Dict[str, np.ndarray]:
    im_left = iio.read(str(src_left))
    im_right = iio.read(str(src_right))

    h_left = adapt_homography_for_shape(im_left.shape, h_left_full)
    h_right = adapt_homography_for_shape(im_right.shape, h_right_full)
    out_shape = compute_output_shape(im_left.shape, h_left)

    valid_left = warp_with_h(np.ones(im_left.shape[:2], dtype=np.float32), h_left, out_shape, order=0) > 0.5
    valid_right = warp_with_h(np.ones(im_right.shape[:2], dtype=np.float32), h_right, out_shape, order=0) > 0.5

    mask_specs = {
        "waterL": (resolve_mask_path(left_stem, roots.water_masks_dir), im_left.shape, h_left, load_keep_mask, 1),
        "waterR": (resolve_mask_path(right_stem, roots.water_masks_dir), im_right.shape, h_right, load_keep_mask, 1),
        "treeL": (resolve_mask_path(left_stem, roots.tree_masks_dir), im_left.shape, h_left, load_keep_mask, 1),
        "treeR": (resolve_mask_path(right_stem, roots.tree_masks_dir), im_right.shape, h_right, load_keep_mask, 1),
        "buildingL": (resolve_mask_path(left_stem, roots.building_masks_dir), im_left.shape, h_left, load_building_positive_mask, 0),
        "buildingR": (resolve_mask_path(right_stem, roots.building_masks_dir), im_right.shape, h_right, load_building_positive_mask, 0),
    }

    rectified_masks = {}
    for key, (path, image_shape, homography, loader, default_value) in mask_specs.items():
        if not path.exists():
            logger.log_missing(f"[MISSING_{key.upper()}] {path}")
        rectified_masks[key] = rectify_optional_mask(
            mask_path=path,
            image_shape=image_shape,
            homography=homography,
            out_shape=out_shape,
            loader=loader,
            default_value=default_value,
        )

    return {
        "Hleft": h_left,
        "Hright": h_right,
        "out_shape": np.array(out_shape, dtype=np.int32),
        "validL": valid_left.astype(np.uint8),
        "validR": valid_right.astype(np.uint8),
        **rectified_masks,
    }


def save_shared_geometry(pair_stem: str, shared: Dict[str, np.ndarray], output_dir: Path, save_preview_png: bool) -> None:
    np.savez(
        output_dir / "homography" / f"{pair_stem}.npz",
        Hleft=shared["Hleft"],
        Hright=shared["Hright"],
        out_shape=shared["out_shape"],
    )

    mask_outputs = {
        "masks/L": shared["validL"],
        "masks/R": shared["validR"],
        "water_masks/L": shared["waterL"],
        "water_masks/R": shared["waterR"],
        "tree_masks/L": shared["treeL"],
        "tree_masks/R": shared["treeR"],
        "building_masks/L": shared["buildingL"],
        "building_masks/R": shared["buildingR"],
    }

    for relative, arr in mask_outputs.items():
        np.save(output_dir / relative / f"{pair_stem}.npy", arr.astype(np.uint8))
        if save_preview_png:
            save_png_uint8(output_dir / "png" / relative / f"{pair_stem}.png", arr.astype(np.uint8) * 255)


def process_pair_images(
    out_stem: str,
    src_left: Path,
    src_right: Path,
    h_left: np.ndarray,
    h_right: np.ndarray,
    out_shape: Tuple[int, int],
    output_dir: Path,
    save_preview_png: bool,
) -> None:
    im_left = iio.read(str(src_left))
    im_right = iio.read(str(src_right))

    warped_left = warp_with_h(im_left, h_left, out_shape, order=1)
    warped_right = warp_with_h(im_right, h_right, out_shape, order=1)

    np.save(output_dir / "L" / f"{out_stem}.npy", warped_left.astype(np.float32))
    np.save(output_dir / "R" / f"{out_stem}.npy", warped_right.astype(np.float32))

    if save_preview_png:
        save_png_uint8(output_dir / "png" / "L" / f"{out_stem}.png", np.clip(warped_left, 0, 255).round().astype(np.uint8))
        save_png_uint8(output_dir / "png" / "R" / f"{out_stem}.png", np.clip(warped_right, 0, 255).round().astype(np.uint8))


def process_split(
    split: SplitConfig,
    roots: Roots,
    max_reference_pairs: int,
    save_preview_png: bool,
) -> None:
    ref_left_dir = split.reference_dir / "L"
    ref_right_dir = split.reference_dir / "R"
    homography_dir = split.reference_dir / "homography"

    ensure_output_dirs(split.output_dir, save_preview_png)
    logger = SplitLogger(split.output_dir)

    ref_left_files = sorted(ref_left_dir.glob("*.iio"))
    if max_reference_pairs > 0:
        ref_left_files = ref_left_files[:max_reference_pairs]

    total = len(ref_left_files)
    started = time.time()
    n_real = 0
    n_augmented = 0
    n_missing = 0
    n_errors = 0

    print(f"\n[{split.name}] reference pairs: {total}")

    for index, ref_left_path in enumerate(ref_left_files, start=1):
        pair_stem = ref_left_path.stem
        ref_right_path = ref_right_dir / ref_left_path.name
        homography_path = homography_dir / f"{pair_stem}.npz"

        if not ref_right_path.exists():
            logger.log_missing(f"[MISSING_REF_R] {pair_stem} | {ref_right_path}")
            n_missing += 1
            continue
        if not homography_path.exists():
            logger.log_missing(f"[MISSING_HOMOGRAPHY] {pair_stem} | {homography_path}")
            n_missing += 1
            continue

        try:
            left_stem, right_stem = parse_pair_stem(pair_stem)

            src_left_real = resolve_source_image(left_stem, roots.track3_root, REAL_EXTS)
            src_right_real = resolve_source_image(right_stem, roots.track3_root, REAL_EXTS)

            missing_real = False
            if not src_left_real.exists():
                logger.log_missing(f"[MISSING_SRC_L_REAL] {pair_stem} | {src_left_real}")
                missing_real = True
            if not src_right_real.exists():
                logger.log_missing(f"[MISSING_SRC_R_REAL] {pair_stem} | {src_right_real}")
                missing_real = True
            if missing_real:
                n_missing += 1
                continue

            homography = np.load(str(homography_path))
            shared = compute_shared_geometry(
                src_left=src_left_real,
                src_right=src_right_real,
                h_left_full=homography["Hleft"],
                h_right_full=homography["Hright"],
                left_stem=left_stem,
                right_stem=right_stem,
                roots=roots,
                logger=logger,
            )
            out_shape = tuple(int(v) for v in shared["out_shape"])

            save_shared_geometry(pair_stem, shared, split.output_dir, save_preview_png)
            process_pair_images(
                out_stem=pair_stem,
                src_left=src_left_real,
                src_right=src_right_real,
                h_left=shared["Hleft"],
                h_right=shared["Hright"],
                out_shape=out_shape,
                output_dir=split.output_dir,
                save_preview_png=save_preview_png,
            )
            n_real += 1

            if split.include_synthetic:
                for job in synthetic_jobs(left_stem, right_stem, roots):
                    process_pair_images(
                        out_stem=str(job["out_stem"]),
                        src_left=job["left_path"],  # type: ignore[arg-type]
                        src_right=job["right_path"],  # type: ignore[arg-type]
                        h_left=shared["Hleft"],
                        h_right=shared["Hright"],
                        out_shape=out_shape,
                        output_dir=split.output_dir,
                        save_preview_png=save_preview_png,
                    )
                    n_augmented += 1

            if index % 100 == 0 or index == total:
                elapsed = (time.time() - started) / 60.0
                print(
                    f"[{split.name}] {index}/{total} | real={n_real} "
                    f"augmented={n_augmented} missing={n_missing} errors={n_errors} "
                    f"elapsed={elapsed:.1f} min"
                )

        except Exception as exc:
            n_errors += 1
            logger.log_error(f"[ERROR] {pair_stem}: {exc}\n{traceback.format_exc()}")
            print(f"[{split.name}] ERROR {pair_stem}: {exc}")

    elapsed = (time.time() - started) / 60.0
    summary = (
        f"split={split.name}\n"
        f"reference_pairs={total}\n"
        f"real_pairs_written={n_real}\n"
        f"synthetic_pairs_written={n_augmented}\n"
        f"missing_pairs={n_missing}\n"
        f"errors={n_errors}\n"
        f"elapsed_minutes={elapsed:.2f}\n"
        f"output_dir={split.output_dir}\n"
    )
    logger.log_summary(summary)
    print(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate rectified training data, masks, and homographies in one run."
    )
    parser.add_argument("--train-reference-dir", type=Path, default=Path("data/synchronic_only"))
    parser.add_argument("--output-root", type=Path, default=Path("data/diachronic-stereo-synthetic"))
    parser.add_argument("--track3-root", type=Path, default=Path("data/Train-Track3-cropped"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("data/Train-Track3-cropped-synthetic"))
    parser.add_argument("--water-masks-dir", type=Path, default=Path("data/water_segmentation/masks"))
    parser.add_argument("--tree-masks-dir", type=Path, default=Path("data/tree_segmentation/masks"))
    parser.add_argument("--building-masks-dir", type=Path, default=Path("data/building_segmentation/masks"))
    parser.add_argument("--max-reference-pairs", type=int, default=0, help="Debug limit. Use 0 for all pairs.")
    parser.add_argument("--no-synthetic", action="store_true", help="Only write real-real training pairs.")
    parser.add_argument("--no-preview-png", action="store_true", help="Skip PNG previews and save only training arrays.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    roots = Roots(
        track3_root=args.track3_root,
        synthetic_root=args.synthetic_root,
        water_masks_dir=args.water_masks_dir,
        tree_masks_dir=args.tree_masks_dir,
        building_masks_dir=args.building_masks_dir,
    )

    process_split(
        split=SplitConfig(
            name="train",
            reference_dir=args.train_reference_dir,
            output_dir=args.output_root / "train",
            include_synthetic=not args.no_synthetic,
        ),
        roots=roots,
        max_reference_pairs=args.max_reference_pairs,
        save_preview_png=not args.no_preview_png,
    )


if __name__ == "__main__":
    main()
