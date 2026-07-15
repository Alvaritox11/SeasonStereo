import json
from pathlib import Path
from typing import List, Optional

from PIL import Image


PROMPTS = {
    "WINTER": (
        "Satellite winter transform. Edit only seasonal appearance. "
        "Preserve exact camera viewpoint, framing, scale, alignment, and scene geometry. "
        "Keep all buildings, roads, field boundaries, landmarks, river banks, shorelines, and water body contours in exactly the same positions, widths, and shapes. "
        "Do not change the size or position of river banks or shorelines. "
        "Apply winter conditions only: snow on rooftops, fields, and open ground; dormant or leafless vegetation; cold muted winter lighting; lower sun angle shadows. "
        "Road network must remain visible and mostly clear of deep snow. Water may appear partially icy. "
        "Allow only minor vehicle position variation. No additions, removals, structural edits, relayout, warping, or perspective change."
    ),
    "SUMMER": (
        "Satellite summer transform. Edit only seasonal appearance. "
        "Preserve exact camera viewpoint, framing, scale, alignment, and scene geometry. "
        "Keep all buildings, roads, field boundaries, landmarks, river banks, shorelines, and water body contours in exactly the same positions, widths, and shapes. "
        "Do not change the size or position of river banks or shorelines. "
        "Apply summer conditions only: lush dense green vegetation, full tree canopies, vibrant healthy fields, bright warm sunlight, and snow-free ground. "
        "Use shorter, tighter shadows consistent with a high summer sun position. "
        "Road network must remain fully visible. Allow only minor vehicle position variation. "
        "No additions, removals, structural edits, relayout, warping, or perspective change."
    ),
    "AUTUMN": (
        "Satellite autumn transform. Edit only seasonal appearance. "
        "Preserve exact camera viewpoint, framing, scale, alignment, and scene geometry. "
        "Keep all buildings, roads, field boundaries, landmarks, river banks, shorelines, and water body contours in exactly the same positions, widths, and shapes. "
        "Do not change the size or position of river banks or shorelines. "
        "Apply autumn conditions only: fall foliage colors, partial leaf loss, muted or harvested fields where appropriate, and softer golden seasonal light. "
        "Do not use exaggerated or highly saturated autumn colors, and avoid any artificial filter-like appearance. "
        "Use longer shadows consistent with a moderate autumn sun angle. "
        "Road network must remain fully visible. Allow only minor vehicle position variation. "
        "No additions, removals, structural edits, relayout, warping, or perspective change."
    ),
    "SPRING": (
        "Satellite spring transform. Edit only seasonal appearance. "
        "Preserve exact camera viewpoint, framing, scale, alignment, and scene geometry. "
        "Keep all buildings, roads, field boundaries, landmarks, river banks, shorelines, and water body contours in exactly the same positions, widths, and shapes. "
        "Do not change the size or position of river banks or shorelines. "
        "Apply spring conditions only: fresh green vegetation, emerging foliage, lush fields, and visible blossoms where appropriate. "
        "Use medium-length shadows consistent with a moderate spring sun angle. "
        "Road network must remain fully visible. Allow only minor vehicle position variation. "
        "No additions, removals, structural edits, relayout, warping, or perspective change."
    ),
}


IMAGE_SIZE = "1K"
ASPECT_RATIO = "1:1"


def default_track_roots(track3_root: Optional[Path] = None) -> List[Path]:
    root = Path(track3_root or "data/Train-Track3-cropped")
    return [root / "Track3-RGB-1", root / "Track3-RGB-2"]


def resolve_track_roots(track3_root: Optional[Path], explicit_roots: Optional[List[Path]]) -> List[Path]:
    if explicit_roots:
        return explicit_roots
    return default_track_roots(track3_root)


def list_aoi_dirs(track_roots: List[Path]) -> List[Path]:
    aoi_dirs: List[Path] = []
    for root in track_roots:
        if not root.exists():
            print(f"Warning: track root does not exist: {root}")
            continue
        for path in sorted(root.iterdir()):
            if path.is_dir():
                aoi_dirs.append(path)
    return aoi_dirs


def find_aoi_dir_by_name(track_roots: List[Path], aoi_name: str) -> Path:
    matches = []
    for root in track_roots:
        candidate = root / aoi_name
        if candidate.is_dir():
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(f"AOI folder not found: {aoi_name}")

    if len(matches) > 1:
        raise ValueError(f"AOI folder appears in multiple track roots: {aoi_name} -> {matches}")

    return matches[0]


def load_aoi_names_from_txt(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        raise FileNotFoundError(f"AOI list file not found: {txt_path}")

    names = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)

    if not names:
        raise ValueError(f"No AOI names found in {txt_path}")

    return names


def get_aoi_dirs_from_txt(track_roots: List[Path], txt_path: Path) -> List[Path]:
    aoi_names = load_aoi_names_from_txt(txt_path)
    return [find_aoi_dir_by_name(track_roots, name) for name in aoi_names]


def choose_first_n_aoi_dirs(track_roots: List[Path], n: int) -> List[Path]:
    all_dirs = list_aoi_dirs(track_roots)
    if len(all_dirs) < n:
        raise ValueError(f"Found only {len(all_dirs)} AOI folders, need {n}.")
    return all_dirs[:n]


def list_images_in_aoi_dir(aoi_dir: Path) -> List[Path]:
    exts = {".tif", ".tiff"}
    return sorted(
        p for p in aoi_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts and "_RGB" in p.stem
    )


def get_images_from_aoi_dirs(aoi_dirs: List[Path]) -> List[Path]:
    all_images: List[Path] = []
    for aoi_dir in aoi_dirs:
        images = list_images_in_aoi_dir(aoi_dir)
        if not images:
            print(f"Warning: no RGB TIFFs found in {aoi_dir}")
            continue
        all_images.extend(images)
    return all_images


def tif_to_temp_png(tif_path: Path, temp_dir: Path) -> Path:
    out_path = temp_dir / f"{tif_path.stem}.png"

    with Image.open(tif_path) as img:
        img = img.convert("RGB")
        if img.size != (1024, 1024):
            raise ValueError(f"Unexpected size for {tif_path}: {img.size}")
        img.save(out_path, format="PNG")

    return out_path


def save_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
