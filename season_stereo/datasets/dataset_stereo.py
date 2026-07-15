import csv
import os
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional

import iio
import numpy as np
import torch
from torch.utils.data import Dataset


CROP_SIZE = 1024


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _parse_bool(x) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    if s == "":
        return None
    raise ValueError(f"Cannot parse boolean value from: {x}")


def _normalize_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    max_v = float(img.max())
    min_v = float(img.min())

    if max_v == 255 and min_v == 0:
        img /= 255.0
    else:
        rng = max_v - min_v
        if rng > 1e-5:
            img -= min_v
            img /= rng
    return img


def _pad_to_size(img: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return img

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 2:
        return np.pad(img, ((top, bottom), (left, right)), mode="edge")
    return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="edge")


def _center_crop(imgs: List[np.ndarray], size: int = CROP_SIZE) -> List[np.ndarray]:
    outputs: List[np.ndarray] = []
    for img in imgs:
        h, w = img.shape[:2]
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        bottom = min(h, top + size)
        right = min(w, left + size)
        cropped = img[top:bottom, left:right]
        cropped = _pad_to_size(cropped, size)
        outputs.append(cropped)
    return outputs


def _to_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _to_2d_map(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    raise ValueError(f"{name} must be HxW or HxWx1, got shape {arr.shape}")


def _load_npz_array(path: str) -> np.ndarray:
    with np.load(path) as data:
        keys = list(data.keys())
        if len(keys) == 1:
            return data[keys[0]]
        if "arr_0" in data:
            return data["arr_0"]
        raise ValueError(f"NPZ file {path} has multiple arrays: {keys}. Cannot choose automatically.")


def _load_array(path: str) -> np.ndarray:
    ext = Path(path).suffix.lower()

    if ext == ".npy":
        return np.load(path)
    if ext == ".npz":
        return _load_npz_array(path)
    if ext in {".iio", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return iio.read(path)

    raise ValueError(f"Unsupported file extension for {path}")


def _base_pair_stem(stem: str) -> str:
    left, right = stem.split("-", 1)
    left_base = left.split("__", 1)[0]
    right_base = right.split("__", 1)[0]
    return f"{left_base}-{right_base}"


def _left_base_stem(stem: str) -> str:
    left, _ = _base_pair_stem(stem).split("-", 1)
    return left


def _sample_aoi(stem: str) -> str:
    # Keeps previous logic: AOI = first 7 characters, e.g. JAX_004
    return _left_base_stem(stem)[:7]


def _load_aois_csv(csv_path: str) -> set[str]:
    aois = set()
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                aois.add(line)
    return aois


def _load_experiment_csv(csv_path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    seen = set()

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain at least a 'filename' column")

        for row in reader:
            filename = row["filename"].strip()
            if not filename:
                continue

            stem = Path(filename).stem
            if "-" not in stem:
                print(f"[WARN] Skipping malformed entry in {csv_path}: {repr(filename)}")
                continue
            
            diachronic = _parse_bool(row.get("diachronic"))

            if stem in seen:
                continue
            seen.add(stem)

            rows.append(
                {
                    "stem": stem,
                    "diachronic": diachronic,
                    "filename_csv": filename,
                }
            )

    return rows


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class StereoDFC(Dataset):
    def __init__(
        self,
        left_dir: str,
        right_dir: str,
        disparity_dir: str,
        experiment_csv: str,
        train: bool = True,
        crop_size: int = CROP_SIZE,
        transforms: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
        left_masks_dir: Optional[str] = None,
        right_masks_dir: Optional[str] = None,
        left_water_masks_dir: Optional[str] = None,
        right_water_masks_dir: Optional[str] = None,
        left_tree_masks_dir: Optional[str] = None,
        right_tree_masks_dir: Optional[str] = None,
        left_building_masks_dir: Optional[str] = None,
        right_building_masks_dir: Optional[str] = None,
        aois_csv: Optional[str] = None,
        image_ext: str = ".npy",
        disparity_ext: str = ".iio",
        mask_ext: str = ".npy",
        return_geometry_masks: bool = False,
        return_water_masks: bool = False,
        mask_out_water: bool = False,
        return_tree_masks: bool = False,
        mask_out_trees: bool = False,
        return_building_masks: bool = False,
    ) -> None:
        super().__init__()

        self.train = train
        self.crop_size = crop_size
        self.transforms = transforms

        self.image_ext = image_ext
        self.disparity_ext = disparity_ext
        self.mask_ext = mask_ext

        self.return_geometry_masks = return_geometry_masks
        self.return_water_masks = return_water_masks
        self.mask_out_water = mask_out_water
        self.return_tree_masks = return_tree_masks
        self.mask_out_trees = mask_out_trees
        self.return_building_masks = return_building_masks

        self.left_dir = os.path.expanduser(left_dir)
        self.right_dir = os.path.expanduser(right_dir)
        self.disp_dir = os.path.expanduser(disparity_dir)

        if left_masks_dir is None or right_masks_dir is None:
            raise ValueError("left_masks_dir and right_masks_dir are mandatory")

        self.left_masks_dir = os.path.expanduser(left_masks_dir)
        self.right_masks_dir = os.path.expanduser(right_masks_dir)

        if not os.path.isdir(self.left_masks_dir):
            raise FileNotFoundError(f"Left masks directory not found: {self.left_masks_dir}")
        if not os.path.isdir(self.right_masks_dir):
            raise FileNotFoundError(f"Right masks directory not found: {self.right_masks_dir}")

        need_water_masks = self.return_water_masks or self.mask_out_water
        self.left_water_masks_dir = None
        self.right_water_masks_dir = None

        if need_water_masks:
            if left_water_masks_dir is None or right_water_masks_dir is None:
                raise ValueError(
                    "Water masks required because return_water_masks=True or mask_out_water=True, "
                    "but left_water_masks_dir/right_water_masks_dir were not provided"
                )

            self.left_water_masks_dir = os.path.expanduser(left_water_masks_dir)
            self.right_water_masks_dir = os.path.expanduser(right_water_masks_dir)

            if not os.path.isdir(self.left_water_masks_dir):
                raise FileNotFoundError(f"Left water masks directory not found: {self.left_water_masks_dir}")
            if not os.path.isdir(self.right_water_masks_dir):
                raise FileNotFoundError(f"Right water masks directory not found: {self.right_water_masks_dir}")

        need_tree_masks = self.return_tree_masks or self.mask_out_trees
        self.left_tree_masks_dir = None
        self.right_tree_masks_dir = None

        if need_tree_masks:
            if left_tree_masks_dir is None or right_tree_masks_dir is None:
                raise ValueError(
                    "Tree masks required because return_tree_masks=True or mask_out_trees=True, "
                    "but left_tree_masks_dir/right_tree_masks_dir were not provided"
                )

            self.left_tree_masks_dir = os.path.expanduser(left_tree_masks_dir)
            self.right_tree_masks_dir = os.path.expanduser(right_tree_masks_dir)

            if not os.path.isdir(self.left_tree_masks_dir):
                raise FileNotFoundError(f"Left tree masks directory not found: {self.left_tree_masks_dir}")
            if not os.path.isdir(self.right_tree_masks_dir):
                raise FileNotFoundError(f"Right tree masks directory not found: {self.right_tree_masks_dir}")

        self.left_building_masks_dir = None
        self.right_building_masks_dir = None

        if self.return_building_masks:
            if left_building_masks_dir is None or right_building_masks_dir is None:
                raise ValueError(
                    "Building masks required because return_building_masks=True, "
                    "but left_building_masks_dir/right_building_masks_dir were not provided"
                )

            self.left_building_masks_dir = os.path.expanduser(left_building_masks_dir)
            self.right_building_masks_dir = os.path.expanduser(right_building_masks_dir)

            if not os.path.isdir(self.left_building_masks_dir):
                raise FileNotFoundError(f"Left building masks directory not found: {self.left_building_masks_dir}")
            if not os.path.isdir(self.right_building_masks_dir):
                raise FileNotFoundError(f"Right building masks directory not found: {self.right_building_masks_dir}")

        all_samples = _load_experiment_csv(experiment_csv)

        if aois_csv is not None:
            aois = _load_aois_csv(aois_csv)
            all_samples = [s for s in all_samples if _sample_aoi(s["stem"]) in aois]
            print(f"Loaded {len(aois)} AOIs from {aois_csv}")

        self.samples = all_samples

        if len(self.samples) == 0:
            raise ValueError("No samples found after applying experiment/AOI filters")

        self.filenames = [s["stem"] for s in self.samples]

        for sample in self.samples:
            stem = sample["stem"]
            base_stem = _base_pair_stem(stem)

            left_file = os.path.join(self.left_dir, stem + self.image_ext)
            right_file = os.path.join(self.right_dir, stem + self.image_ext)
            disp_file = os.path.join(self.disp_dir, base_stem + self.disparity_ext)
            left_mask_file = os.path.join(self.left_masks_dir, base_stem + self.mask_ext)
            right_mask_file = os.path.join(self.right_masks_dir, base_stem + self.mask_ext)
            left_real_file  = os.path.join(self.left_dir,  base_stem + self.image_ext)
            right_real_file = os.path.join(self.right_dir, base_stem + self.image_ext)

            if not os.path.exists(left_file):
                raise FileNotFoundError(f"{left_file} not found")
            if not os.path.exists(right_file):
                raise FileNotFoundError(f"{right_file} not found")
            if not os.path.exists(disp_file):
                raise FileNotFoundError(f"{disp_file} not found")
            if not os.path.exists(left_mask_file):
                raise FileNotFoundError(f"{left_mask_file} not found")
            if not os.path.exists(right_mask_file):
                raise FileNotFoundError(f"{right_mask_file} not found")
            if not os.path.exists(left_real_file):
                raise FileNotFoundError(
                    f"Real left image not found: {left_real_file}\n"
                    f"  (required for photometric loss — diachronic pair {stem})"
                )
            if not os.path.exists(right_real_file):
                raise FileNotFoundError(
                    f"Real right image not found: {right_real_file}\n"
                    f"  (required for photometric loss — diachronic pair {stem})"
                )


            if need_water_masks:
                left_water_file = os.path.join(self.left_water_masks_dir, base_stem + self.mask_ext)
                right_water_file = os.path.join(self.right_water_masks_dir, base_stem + self.mask_ext)

                if not os.path.exists(left_water_file):
                    raise FileNotFoundError(f"{left_water_file} not found")
                if not os.path.exists(right_water_file):
                    raise FileNotFoundError(f"{right_water_file} not found")
                
            if need_tree_masks:
                left_tree_file = os.path.join(self.left_tree_masks_dir, base_stem + self.mask_ext)
                right_tree_file = os.path.join(self.right_tree_masks_dir, base_stem + self.mask_ext)

                if not os.path.exists(left_tree_file):
                    raise FileNotFoundError(f"{left_tree_file} not found")
                if not os.path.exists(right_tree_file):
                    raise FileNotFoundError(f"{right_tree_file} not found")

            if self.return_building_masks:
                left_building_file = os.path.join(self.left_building_masks_dir, base_stem + self.mask_ext)
                right_building_file = os.path.join(self.right_building_masks_dir, base_stem + self.mask_ext)

                if not os.path.exists(left_building_file):
                    raise FileNotFoundError(f"{left_building_file} not found")
                if not os.path.exists(right_building_file):
                    raise FileNotFoundError(f"{right_building_file} not found")

        print(f"Dataset ready: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.samples[idx]
        stem = meta["stem"]
        base_stem = _base_pair_stem(stem)

        left = _load_array(os.path.join(self.left_dir, stem + self.image_ext))
        right = _load_array(os.path.join(self.right_dir, stem + self.image_ext))
        left_real  = _load_array(os.path.join(self.left_dir,  base_stem + self.image_ext))
        right_real = _load_array(os.path.join(self.right_dir, base_stem + self.image_ext))

        disp = _to_2d_map(
            _load_array(os.path.join(self.disp_dir, base_stem + self.disparity_ext)),
            "disparity",
        )

        valid = np.isfinite(disp) & (disp > 0)
        disp = np.nan_to_num(disp, copy=False).astype(np.float32, copy=False)

        left_geom_mask = _to_bool_mask(_load_array(os.path.join(self.left_masks_dir, base_stem + self.mask_ext)))
        right_geom_mask = _to_bool_mask(_load_array(os.path.join(self.right_masks_dir, base_stem + self.mask_ext)))

        if left_geom_mask.shape[:2] != disp.shape[:2]:
            raise ValueError(
                f"Left geometry mask shape {left_geom_mask.shape} does not match disparity shape {disp.shape} for {stem}"
            )
        if right_geom_mask.shape[:2] != disp.shape[:2]:
            raise ValueError(
                f"Right geometry mask shape {right_geom_mask.shape} does not match disparity shape {disp.shape} for {stem}"
            )

        # Geometry mask is always mandatory
        valid = valid & left_geom_mask

        left_water_mask = None
        right_water_mask = None

        if self.return_water_masks or self.mask_out_water:
            left_water_mask = _to_bool_mask(
                _load_array(os.path.join(self.left_water_masks_dir, base_stem + self.mask_ext))
            )
            right_water_mask = _to_bool_mask(
                _load_array(os.path.join(self.right_water_masks_dir, base_stem + self.mask_ext))
            )

            if left_water_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Left water mask shape {left_water_mask.shape} does not match disparity shape {disp.shape} for {stem}"
                )
            if right_water_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Right water mask shape {right_water_mask.shape} does not match disparity shape {disp.shape} for {stem}"
                )

            # 1 = keep pixel, 0 = ignore water
            if self.mask_out_water:
                valid = valid & left_water_mask

        # Tree masks
        left_tree_mask = None
        right_tree_mask = None

        if self.return_tree_masks or self.mask_out_trees:
            left_tree_mask = _to_bool_mask(
                _load_array(os.path.join(self.left_tree_masks_dir, base_stem + self.mask_ext))
            )
            right_tree_mask = _to_bool_mask(
                _load_array(os.path.join(self.right_tree_masks_dir, base_stem + self.mask_ext))
            )

            if left_tree_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Left tree mask shape {left_tree_mask.shape} does not match "
                    f"disparity shape {disp.shape} for {stem}"
                )
            if right_tree_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Right tree mask shape {right_tree_mask.shape} does not match "
                    f"disparity shape {disp.shape} for {stem}"
                )
            # 1 = keep pixel (not tree), 0 = tree (exclude)
            if self.mask_out_trees:
                valid = valid & left_tree_mask

        left_building_mask = None
        right_building_mask = None

        if self.return_building_masks:
            left_building_mask = _to_bool_mask(
                _load_array(os.path.join(self.left_building_masks_dir, base_stem + self.mask_ext))
            )
            right_building_mask = _to_bool_mask(
                _load_array(os.path.join(self.right_building_masks_dir, base_stem + self.mask_ext))
            )

            if left_building_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Left building mask shape {left_building_mask.shape} does not match "
                    f"disparity shape {disp.shape} for {stem}"
                )
            if right_building_mask.shape[:2] != disp.shape[:2]:
                raise ValueError(
                    f"Right building mask shape {right_building_mask.shape} does not match "
                    f"disparity shape {disp.shape} for {stem}"
                )

        left = _normalize_image(left)
        right = _normalize_image(right)

        left_real  = _normalize_image(left_real)
        right_real = _normalize_image(right_real)

        sample_np = {
            "left": left,
            "right": right,
            "left_real": left_real,
            "right_real": right_real,
            "disparity": disp,
            "valid": valid.astype(np.float32, copy=False),
        }

        if self.return_geometry_masks:
            sample_np["left_mask"] = left_geom_mask.astype(np.float32, copy=False)
            sample_np["right_mask"] = right_geom_mask.astype(np.float32, copy=False)

        if self.return_water_masks:
            sample_np["left_water_mask"] = left_water_mask.astype(np.float32, copy=False)
            sample_np["right_water_mask"] = right_water_mask.astype(np.float32, copy=False)

        if self.return_tree_masks:
            sample_np["left_tree_mask"] = left_tree_mask.astype(np.float32, copy=False)
            sample_np["right_tree_mask"] = right_tree_mask.astype(np.float32, copy=False)

        if self.return_building_masks:
            sample_np["left_building_mask"] = left_building_mask.astype(np.float32, copy=False)
            sample_np["right_building_mask"] = right_building_mask.astype(np.float32, copy=False)
            
        if not self.train:
            cropped = _center_crop(list(sample_np.values()), self.crop_size)
            for key, arr in zip(sample_np.keys(), cropped):
                sample_np[key] = arr

        tensor_sample: Dict[str, torch.Tensor] = {}
        for k, arr in sample_np.items():
            if arr.ndim == 2:
                tensor = torch.from_numpy(arr).unsqueeze(0)
            else:
                tensor = torch.from_numpy(arr.transpose(2, 0, 1))
            tensor_sample[k] = tensor.float()

        if self.transforms is not None and self.train:
            tensor_sample = self.transforms(tensor_sample)

        if meta["diachronic"] is not None:
            tensor_sample["diachronic"] = torch.tensor(float(bool(meta["diachronic"])), dtype=torch.float32)

        tensor_sample["filename"] = stem
        tensor_sample["base_filename"] = base_stem
        return tensor_sample
