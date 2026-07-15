#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from scipy import ndimage as ndi

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

BACKBONE = "nvidia/mit-b2"
NUM_CLASSES = 9

ID2LABEL = {
    0: "Background",
    1: "Bareland",
    2: "Grass",
    3: "Pavement",
    4: "Road",
    5: "Tree",
    6: "Water",
    7: "Cropland",
    8: "Building",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Root folder with images")
    parser.add_argument("--output", type=str, required=True, help="Root output folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to local .pt file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional probability threshold for tree mask. If omitted, use argmax.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. Use 0 for all images.")

    # For trees: only remove very tiny isolated detections
    parser.add_argument("--min-tree-component-pct", type=float, default=0.05,
                        help="Remove tree connected components smaller than this percent of image area.")
    parser.add_argument("--fill-hole-pct", type=float, default=0.0,
                        help="Fill non-tree holes inside tree regions smaller than this percent of image area. "
                             "Default 0 means disabled.")
    return parser.parse_args()


def list_images(root: Path):
    if root.is_file():
        return [root]
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def preprocess(image_pil: Image.Image):
    image = image_pil.resize((1000, 1000), resample=Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float().unsqueeze(0)


def find_tree_index(id2label):
    normalized = {int(k): v for k, v in id2label.items()}
    for idx, label in normalized.items():
        if "tree" in label.lower():
            return idx
    raise RuntimeError(f"Could not find tree class in id2label: {normalized}")


def clean_state_dict(sd):
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("model.", "")
        out[k] = v
    return out


def build_output_paths(img_path: Path, masks_root: Path, probs_root: Path):
    stem = img_path.stem
    masks_root.mkdir(parents=True, exist_ok=True)
    probs_root.mkdir(parents=True, exist_ok=True)

    mask_path = masks_root / f"{stem}_mask.png"
    prob_path = probs_root / f"{stem}_tree_prob.png"
    return mask_path, prob_path


def remove_small_components(binary_mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return binary_mask

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num = ndi.label(binary_mask, structure=structure)
    if num == 0:
        return binary_mask

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_area_px
    keep[0] = False

    cleaned = keep[labeled]
    return cleaned


def fill_small_holes(binary_mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    if max_hole_area_px <= 0:
        return binary_mask

    inverse = ~binary_mask
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, num = ndi.label(inverse, structure=structure)
    if num == 0:
        return binary_mask

    counts = np.bincount(labeled.ravel())
    border_labels = np.unique(
        np.concatenate([
            labeled[0, :],
            labeled[-1, :],
            labeled[:, 0],
            labeled[:, -1],
        ])
    )

    cleaned = binary_mask.copy()
    for lab in range(1, num + 1):
        if lab in border_labels:
            continue
        if counts[lab] < max_hole_area_px:
            cleaned[labeled == lab] = True

    return cleaned


def postprocess_tree_mask(tree_mask: np.ndarray,
                          min_tree_component_pct: float,
                          fill_hole_pct: float) -> np.ndarray:
    total_pixels = tree_mask.size
    min_area_px = int(round((min_tree_component_pct / 100.0) * total_pixels))
    max_hole_area_px = int(round((fill_hole_pct / 100.0) * total_pixels))

    cleaned = remove_small_components(tree_mask, min_area_px=min_area_px)

    if fill_hole_pct > 0:
        cleaned = fill_small_holes(cleaned, max_hole_area_px=max_hole_area_px)

    return cleaned


def main():
    args = parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    masks_root = output_root / "masks"
    probs_root = output_root / "probs"
    stats_path = output_root / "tree_percentages.txt"

    masks_root.mkdir(parents=True, exist_ok=True)
    probs_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    config = SegformerConfig.from_pretrained(
        BACKBONE,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = SegformerForSemanticSegmentation(config)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = clean_state_dict(ckpt["model_state_dict"])

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    model.to(device)
    model.eval()

    tree_idx = find_tree_index(model.config.id2label)
    print("Tree class index:", tree_idx)

    images = list_images(input_root)
    if not images:
        raise RuntimeError(f"No images found in {input_root}")

    print(f"Found {len(images)} images")
    if args.limit > 0:
        images = images[:args.limit]
        print(f"Limited to {len(images)} images")

    with open(stats_path, "w") as f_stats:
        f_stats.write("image_name tree_percentage tree_pixels total_pixels\n")

        for i, img_path in enumerate(images, 1):
            image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = image.size

            pixel_values = preprocess(image).to(device)

            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits

            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0]

            probs = torch.softmax(logits, dim=0)
            pred = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)
            tree_prob = probs[tree_idx].cpu().numpy()

            if args.threshold is None:
                raw_tree_mask = (pred == tree_idx)
            else:
                raw_tree_mask = (tree_prob >= args.threshold)

            tree_mask = postprocess_tree_mask(
                raw_tree_mask,
                min_tree_component_pct=args.min_tree_component_pct,
                fill_hole_pct=args.fill_hole_pct,
            )

            tree_pixels = int(tree_mask.sum())
            total_pixels = int(tree_mask.size)
            tree_percentage = 100.0 * tree_pixels / total_pixels

            # Validation mask: trees = 0, rest = 1
            final_mask = (~tree_mask).astype(np.uint8)

            mask_path, prob_path = build_output_paths(
                img_path=img_path,
                masks_root=masks_root,
                probs_root=probs_root,
            )

            Image.fromarray(final_mask).save(mask_path)

            prob_u8 = np.clip(tree_prob * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(prob_u8).save(prob_path)

            f_stats.write(
                f"{img_path.name} {tree_percentage:.6f} {tree_pixels} {total_pixels}\n"
            )

            print(
                f"[{i}/{len(images)}] {img_path.name} -> {mask_path.name} | "
                f"tree={tree_percentage:.4f}%"
            )

    print(f"Saved tree percentages to: {stats_path}")
    print("Done.")


if __name__ == "__main__":
    main()
