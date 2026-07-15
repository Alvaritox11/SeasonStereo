import os
import csv
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import thirdparty


# --------------------------- Reproducibility ---------------------------

def set_reproducible(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- Data model ---------------------------

@dataclass(frozen=True)
class PairSample:
    pair_id: str
    left_path: Path
    right_path: Path

    def out_dir(self, root: Path, model_name: str) -> Path:
        return root / model_name

    def out_disp_path(self, root: Path, model_name: str) -> Path:
        return self.out_dir(root, model_name) / f"{self.pair_id}.iio"


# --------------------------- I/O helpers ---------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_image_any(path: Path) -> torch.Tensor:
    """
    Load image -> [1,3,H,W] float in [0,1]
    """
    arr = iio.read(str(path))

    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)

    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(-1)

    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]

    if arr.max() == 255:
        arr = arr.astype(np.float32) / 255.0
    else:
        m, M = float(arr.min()), float(arr.max())
        arr = (arr - m) / (M - m + 1e-12)

    t = torch.from_numpy(arr).permute(2, 0, 1).float()
    return t.unsqueeze(0)


def pad_to_multiple(
    x: torch.Tensor, multiple: int = 32
) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    l, r, t, b = pad
    return x[..., t : x.shape[-2] - b, l : x.shape[-1] - r]


def center_crop_if_large(
    imgL: torch.Tensor,
    imgR: torch.Tensor,
    max_side: int = 1024,
):
    H, W = imgL.shape[-2:]
    if max_side <= 0 or (H <= max_side and W <= max_side):
        return imgL, imgR, None

    new_H = min(H, max_side)
    new_W = min(W, max_side)
    h0 = (H - new_H) // 2
    w0 = (W - new_W) // 2
    h1, w1 = h0 + new_H, w0 + new_W

    return imgL[..., h0:h1, w0:w1], imgR[..., h0:h1, w0:w1], (h0, h1, w0, w1)


def _collate_minibatch(samples: List[PairSample], max_side: int, device: str):
    imgs_L, imgs_R, metas = [], [], []

    for s in samples:
        L = _read_image_any(s.left_path)
        R = _read_image_any(s.right_path)
        Lc, Rc, crop = center_crop_if_large(L, R, max_side=max_side)
        imgs_L.append(Lc)
        imgs_R.append(Rc)
        metas.append({"pair": s, "crop": crop})

    Hmax = max(t.shape[-2] for t in imgs_L)
    Wmax = max(t.shape[-1] for t in imgs_L)

    def _pad_to(Hb, Wb, x):
        ph = Hb - x.shape[-2]
        pw = Wb - x.shape[-1]
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        return F.pad(x, pad, mode="replicate"), pad

    out_L, out_R = [], []
    for i, (L, R) in enumerate(zip(imgs_L, imgs_R)):
        Lp, pad = _pad_to(Hmax, Wmax, L)
        Rp, _ = _pad_to(Hmax, Wmax, R)
        metas[i]["pad"] = pad
        out_L.append(Lp)
        out_R.append(Rp)

    batch_L = torch.cat(out_L, dim=0).to(device, non_blocking=True)
    batch_R = torch.cat(out_R, dim=0).to(device, non_blocking=True)
    return batch_L, batch_R, metas


# --------------------------- Predictors ---------------------------

class MonSterPredictor:
    def __init__(self, ckpt: str, depth_anything_v2_path: str, device: str):
        self.device = device
        self.model = thirdparty.build_monster(
            monster_ckpt=ckpt,
            depth_anything_v2_path=depth_anything_v2_path,
            device=device,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        L255, R255 = batch_L * 255.0, batch_R * 255.0
        Lp, pad = pad_to_multiple(L255, 32)
        Rp, _ = pad_to_multiple(R255, 32)
        disp = self.model(Lp, Rp, iters=32, test_mode=True)
        disp = unpad(disp, pad).squeeze(1).cpu().numpy()
        return disp


class FoundationStereoPredictor:
    def __init__(self, ckpt: str, device: str, scale: float = 1.0):
        self.device = device
        self.scale = float(scale)

        if not (0 < self.scale <= 1.0):
            raise ValueError("scale must be in (0, 1]")

        self.stereo_model = thirdparty.build_foundation_stereo(
            foundation_ckpt=ckpt,
            device=device,
        )
        self.stereo_model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor, batch_R: torch.Tensor) -> np.ndarray:
        """
        batch_L, batch_R: [B,3,H,W] in [0,1]
        """
        B, _, H, W = batch_L.shape

        if self.scale != 1.0:
            new_H = int(H * self.scale)
            new_W = int(W * self.scale)

            batch_L = torch.nn.functional.interpolate(
                batch_L, size=(new_H, new_W), mode="bilinear", align_corners=False
            )
            batch_R = torch.nn.functional.interpolate(
                batch_R, size=(new_H, new_W), mode="bilinear", align_corners=False
            )

        L = batch_L.to(self.device) * 255.0
        R = batch_R.to(self.device) * 255.0

        padder = thirdparty.FsInputPadder(L.shape, divis_by=32, force_square=False)
        Lp, Rp = padder.pad(L, R)

        with torch.autocast(
            device_type="cuda",
            enabled=str(self.device).startswith("cuda"),
        ):
            disp = self.stereo_model.run_hierachical(
                Lp, Rp,
                iters=32,
                test_mode=True,
                small_ratio=0.5
            )

        disp = padder.unpad(disp).squeeze(1)  # [B,Hs,Ws]

        if self.scale != 1.0:
            disp = torch.nn.functional.interpolate(
                disp.unsqueeze(1),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            # IMPORTANT: rescale disparity
            disp = disp / self.scale

        return disp.cpu().numpy().astype(np.float32)


# --------------------------- Pair list ---------------------------

def read_pairs_list(pairs_list: str) -> List[PairSample]:
    samples: List[PairSample] = []

    with open(pairs_list, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line or line.startswith("#") or line.startswith("left,right"):
                continue

            parts = line.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_idx} in {pairs_list} must have exactly 2 columns: left,right"
                )

            left_path, right_path = parts
            sample_id = Path(left_path).stem

            samples.append(
                PairSample(
                    pair_id=sample_id,
                    left_path=Path(left_path),
                    right_path=Path(right_path),
                )
            )

    return samples


# --------------------------- Runner ---------------------------

def run_batches(
    samples: List[PairSample],
    predictor,
    out_root: Path,
    model_name: str,
    batch_size: int = 1,
    max_side: int = 1024,
    overwrite: bool = False,
):
    device = predictor.device

    for i in tqdm(
        range(0, len(samples), batch_size),
        total=(len(samples) + batch_size - 1) // batch_size,
        desc=f"Predicting {model_name}",
    ):
        chunk = samples[i : i + batch_size]

        if not overwrite:
            chunk = [
                s for s in chunk
                if not s.out_disp_path(out_root, model_name).exists()
            ]
            if not chunk:
                continue

        batch_L, batch_R, metas = _collate_minibatch(
            chunk, max_side=max_side, device=device
        )
        disp_np = predictor.predict_batch(batch_L, batch_R)

        for bi, meta in enumerate(metas):
            s = meta["pair"]
            lpad, rpad, tpad, bpad = meta["pad"]

            disp_full = disp_np[bi]
            disp_unpadded = disp_full[
                tpad : disp_full.shape[0] - bpad,
                lpad : disp_full.shape[1] - rpad,
            ].astype(np.float32)

            pair_dir = s.out_dir(out_root, model_name)
            out_path = s.out_disp_path(out_root, model_name)
            _ensure_dir(pair_dir)
            iio.write(str(out_path), disp_unpadded)


# --------------------------- CLI ---------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_list", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model", type=str, required=True,
                        choices=["monster", "monster++", "foundationstereo"])
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--depth_anything_v2_path", type=str, default=None,
                        help="Path to the Depth-Anything V2 checkpoint used by MonSter.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--max_side",
        type=int,
        default=1024,
        help="Center-crop images larger than this side length. Use 0 to preserve full image size.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Debug limit. Use 0 for all pairs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_reproducible(42)
    torch.autograd.set_grad_enabled(False)

    samples = read_pairs_list(args.pairs_list)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Found {len(samples)} pairs")

    model_key = args.model.lower()

    if model_key in ("monster", "monster++"):
        if args.depth_anything_v2_path is None:
            raise ValueError("--depth_anything_v2_path is required for MonSter / MonSter++")
        predictor = MonSterPredictor(
            ckpt=args.ckpt_path,
            depth_anything_v2_path=args.depth_anything_v2_path,
            device=args.device,
        )
    elif model_key == "foundationstereo":
        predictor = FoundationStereoPredictor(
            ckpt=args.ckpt_path,
            device=args.device,
            scale=args.scale
        )
    else:
        raise RuntimeError(f"Unknown model: {args.model}")

    run_batches(
        samples=samples,
        predictor=predictor,
        out_root=Path(args.out_dir),
        model_name=model_key,
        batch_size=args.batch_size,
        max_side=args.max_side,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
