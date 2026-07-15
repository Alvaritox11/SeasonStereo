"""
Evaluate stereo model checkpoints on all test sets and compute altitude MAE.

Three dataset adapters — same pattern as the original evaluate_all.py:
  discover_omaha()        — OMA sync / diach  (flat RPC lookup)
  discover_jax()          — JAX               (per-AOI RPC lookup)
  discover_buenos_aires() — IARPA 001/002/003 (per-IARPA RPC lookup, long WV names)

Usage:
  python eval_test.py \\
    --models \\
        baseline:checkpoints/monster++-mix_all.pth \\
        season:checkpoints/season-stereo-final.pth \\
    --test-root  data/diachronic-stereo-synthetic/test \\
    --output-dir outputs/test_eval \\
    --datasets omaha_sync omaha_diach jax buenos_aires \\
    --device cuda:0 \\
    --skip-existing
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import misc
import thirdparty


# ─────────────────────────────────────────────────────────────────────────────
# Relative RPC + DSM roots under --test-root
# ─────────────────────────────────────────────────────────────────────────────

# Omaha: flat RPC dir, OMA+JAX DSMs
OMA_JAX_RPC_REL = Path("OMA") / "all_oma_and_jax_dsm_and_rpc" / "root_dir"
OMA_JAX_DSM_REL = Path("OMA") / "all_oma_and_jax_dsm_and_rpc" / "Track3-Truth"

# JAX: per-AOI RPC dir, per-AOI DSMs
JAX_RPC_REL     = Path("JAX") / "test_jax_dsm_and_rpc" / "root_dir" / "crops_rpcs_ba_v2"
JAX_DSM_REL     = Path("JAX") / "test_jax_dsm_and_rpc" / "Track3-Truth"

# Buenos Aires: per-IARPA RPC dir, per-IARPA DSMs
BA_RPC_REL      = Path("IARPA") / "test_buenos_aires_dsm_and_rpc" / "root_dir" / "rpcs_ba"
BA_DSM_REL      = Path("IARPA") / "test_buenos_aires_dsm_and_rpc" / "Truth"


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_image_any(path: Path) -> torch.Tensor:
    """Load any image → (1, 3, H, W) float in [0, 1]."""
    arr = iio.read(str(path)).astype(np.float32)
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=0.0)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr.squeeze(-1)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    if arr.max() > 1.0:
        arr = arr / 255.0
    else:
        m, M = float(arr.min()), float(arr.max())
        arr = (arr - m) / (M - m + 1e-12)
    return torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0)


def pad_to_multiple(x: torch.Tensor, multiple: int = 32):
    h, w = x.shape[-2:]
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(x, pad, mode="replicate"), pad


def unpad(x: torch.Tensor, pad) -> torch.Tensor:
    l, r, t, b = pad
    return x[..., t: x.shape[-2] - b, l: x.shape[-1] - r]


def center_crop_if_large(imgL, imgR, max_side: int = 1024):
    H, W = imgL.shape[-2:]
    if H <= max_side and W <= max_side:
        return imgL, imgR, None
    new_H, new_W = min(H, max_side), min(W, max_side)
    h0 = (H - new_H) // 2
    w0 = (W - new_W) // 2
    return imgL[..., h0:h0+new_H, w0:w0+new_W], \
           imgR[..., h0:h0+new_H, w0:w0+new_W], \
           (h0, h0+new_H, w0, w0+new_W)


def trim_border_disparity(disp, H_left, H_right, k: int = 32):
    h, w = disp.shape[:2]
    out = disp[k: h-k, k: w-k]
    if H_left is None and H_right is None:
        return out, None, None
    T = np.array([[1, 0, -float(k)], [0, 1, -float(k)], [0, 0, 1]], dtype=float)
    return out, T @ H_left, T @ H_right


def _to_png(t3hw: torch.Tensor) -> np.ndarray:
    return (t3hw * 255.0).clip(0, 255).byte().cpu().numpy().transpose(1, 2, 0)


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_COLS = [
    "model", "dataset", "pair_id", "aoi",
    "left_path", "right_path", "homography_path",
    "left_rpc_json", "right_rpc_json",
    "gt_dsm_path", "pred_dsm_path",
    "mae", "mae_filtered", "mae_no_water",
    "status", "error",
]


def _summary_path(out_dir: Path) -> Path:
    return out_dir / "summary.csv"


def _init_summary(out_dir: Path) -> None:
    p = _summary_path(out_dir)
    if not p.exists():
        _ensure_dir(out_dir)
        with open(p, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=SUMMARY_COLS).writeheader()


def _append_summary(out_dir: Path, row: dict) -> None:
    with open(_summary_path(out_dir), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writerow({k: row.get(k, "") for k in SUMMARY_COLS})


def _load_done_pairs(out_dir: Path) -> Set[Tuple[str, str]]:
    """Return set of (model, pair_id) already marked 'ok' in summary.csv."""
    done: Set[Tuple[str, str]] = set()
    p = _summary_path(out_dir)
    if not p.exists():
        return done
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok":
                done.add((row["model"], row["pair_id"]))
    return done


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PairSample:
    dataset:          str
    pair_id:          str
    aoi:              str
    left_path:        Path
    right_path:       Path
    out_dir:          Path           # model-level output dir
    homography_path:  Optional[Path] = None
    left_rpc_json:    Optional[Path] = None
    right_rpc_json:   Optional[Path] = None
    gt_dsm_path:      Optional[Path] = None

    def pair_outdir(self) -> Path:
        # Buenos Aires pairs have AOI prefix in pair_id (e.g. "IARPA_001/xxxxx")
        safe_id = self.pair_id.replace("/", os.sep)
        return self.out_dir / self.dataset / safe_id


# ─────────────────────────────────────────────────────────────────────────────
# Model predictor
# ─────────────────────────────────────────────────────────────────────────────

class MonSterPredictor:
    def __init__(self, ckpt: str, depth_anything_v2_path: str, device: str):
        self.device = device
        self.model = thirdparty.build_monster(
            monster_ckpt=ckpt,
            depth_anything_v2_path=depth_anything_v2_path,
            device=device,
            eval_only=True,
        )
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, batch_L: torch.Tensor,
                      batch_R: torch.Tensor) -> np.ndarray:
        L255, R255 = batch_L * 255.0, batch_R * 255.0
        Lp, pad    = pad_to_multiple(L255, 32)
        Rp, _      = pad_to_multiple(R255, 32)
        disp = self.model(Lp, Rp, iters=32, test_mode=True)
        return unpad(disp, pad).squeeze(1).cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Batch collation
# ─────────────────────────────────────────────────────────────────────────────

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
        ph, pw = Hb - x.shape[-2], Wb - x.shape[-1]
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        return F.pad(x, pad, mode="replicate"), pad

    out_L, out_R = [], []
    for i, (L, R) in enumerate(zip(imgs_L, imgs_R)):
        Lp, pad = _pad_to(Hmax, Wmax, L)
        Rp, _   = _pad_to(Hmax, Wmax, R)
        metas[i]["pad"] = pad
        out_L.append(Lp)
        out_R.append(Rp)

    batch_L = torch.cat(out_L, dim=0).to(device, non_blocking=True)
    batch_R = torch.cat(out_R, dim=0).to(device, non_blocking=True)
    return batch_L, batch_R, metas


# ─────────────────────────────────────────────────────────────────────────────
# Per-pair output + DSM triangulation
# ─────────────────────────────────────────────────────────────────────────────

def compute_disparity_mask(disp: np.ndarray) -> np.ndarray:
    """
    Valid-disparity boolean mask.
    Rejects: non-finite, non-positive, and pixels whose right-image
    correspondence (x - d) falls outside the image width.
    This cleanly removes no-coverage zones from non-overlapping image pairs.
    """
    H, W = disp.shape
    cols       = np.arange(W, dtype=float)[np.newaxis, :]   # (1, W) broadcast
    right_cols = cols - disp                                  # xR = x − d
    return (
        np.isfinite(disp)
        & (disp > 0.0)
        & (right_cols >= 0.0)
        & (right_cols < float(W))
    )


def _save_pair_outputs_and_row(meta: dict, disp: np.ndarray,
                                imgL_3hw: torch.Tensor, imgR_3hw: torch.Tensor,
                                border_trim_k: int, model_name: str) -> dict:
    s: PairSample = meta["pair"]
    crop = meta["crop"]
    pair_dir = s.pair_outdir()
    _ensure_dir(pair_dir)

    # ── Always save: images + disparity ───────────────────────────────────
    iio.write(str(pair_dir / "left_image.png"),  _to_png(imgL_3hw))
    iio.write(str(pair_dir / "right_image.png"), _to_png(imgR_3hw))
    iio.write(str(pair_dir / "disparity.iio"),   disp.astype(np.float32))

    row = dict(
        model          = model_name,
        dataset        = s.dataset,
        pair_id        = s.pair_id,
        aoi            = s.aoi,
        left_path      = str(s.left_path),
        right_path     = str(s.right_path),
        homography_path= str(s.homography_path) if s.homography_path else "",
        left_rpc_json  = str(s.left_rpc_json)   if s.left_rpc_json   else "",
        right_rpc_json = str(s.right_rpc_json)   if s.right_rpc_json   else "",
        gt_dsm_path    = str(s.gt_dsm_path)      if s.gt_dsm_path      else "",
        pred_dsm_path  = "",
        mae            = "", mae_filtered = "", mae_no_water = "",
        status         = "skipped", error = "",
    )

    if not (s.homography_path and s.left_rpc_json and
            s.right_rpc_json  and s.gt_dsm_path):
        return row

    # ── Triangulation ─────────────────────────────────────────────────────
    try:
        hom    = np.load(str(s.homography_path))
        Hleft  = hom["Hleft"]
        Hright = hom["Hright"]

        # Adjust homographies for any center-crop applied before inference
        if crop is not None:
            h0, h1, w0, w1 = crop
            T_crop = np.array([[1,0,-float(w0)],[0,1,-float(h0)],[0,0,1]], dtype=float)
            Hleft  = T_crop @ Hleft
            Hright = T_crop @ Hright

        left_rpc  = misc.rpc_from_json(str(s.left_rpc_json))
        right_rpc = misc.rpc_from_json(str(s.right_rpc_json))

        with rasterio.open(str(s.gt_dsm_path)) as src:
            dsm_meta  = src.meta.copy()
            gt_dsm_np = src.read(1).astype(np.float32)
            dsm_shape = gt_dsm_np.shape
        
        # finite = gt_dsm_np[np.isfinite(gt_dsm_np)]
        # alt_min = float(np.percentile(finite, 2))  if finite.size else 0
        # alt_max = float(np.percentile(finite, 98)) if finite.size else 1
        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.imshow(gt_dsm_np, cmap="terrain", vmin=alt_min, vmax=alt_max)
        # ax.set_title("GT DSM")
        # ax.axis("off")
        # fig.savefig(str(pair_dir / "gt_dsm.png"), dpi=100, bbox_inches="tight")
        # plt.close(fig)

        valid_mask_full = compute_disparity_mask(disp)
        disp_valid_full = disp.astype(np.float32).copy()
        disp_valid_full[~valid_mask_full] = np.nan
        iio.write(str(pair_dir / "disparity_masked.iio"), disp_valid_full)

        disp_valid, Hl2, Hr2 = trim_border_disparity(
            disp_valid_full, Hleft, Hright, k=border_trim_k
        )
        valid_mask = compute_disparity_mask(disp_valid)
        disp_valid = disp_valid.astype(np.float32).copy()
        disp_valid[~valid_mask] = np.nan

        # Use masked disparity for all downstream steps
        alt_img = misc.altitude_image_from_disparity_vectorized(
            disp_valid, left_rpc, right_rpc, Hl2, Hr2 
        )
        pred_dsm_np, pred_meta = misc.rectified_altitude_to_dsm(
            alt_img, left_rpc, Hl2, dsm_shape, dsm_meta
        )

        # Save predicted DSM as GeoTIFF
        pred_path = pair_dir / "pred_dsm.tif"
        with rasterio.open(
            str(pred_path), "w", driver="GTiff",
            height=pred_dsm_np.shape[0], width=pred_dsm_np.shape[1],
            count=1, dtype=pred_meta["dtype"], crs=pred_meta["crs"],
            transform=pred_meta["transform"], nodata=pred_meta["nodata"],
        ) as dst:
            dst.write(pred_dsm_np, 1)

        # fig, ax = plt.subplots(figsize=(6, 5))
        # ax.imshow(pred_dsm_np, cmap="terrain", vmin=alt_min, vmax=alt_max)
        # ax.set_title("Predicted DSM")
        # ax.axis("off")
        # fig.savefig(str(pair_dir / "pred_dsm.png"), dpi=100, bbox_inches="tight")
        # plt.close(fig)

        # Compute MAE (three masking modes)
        gt_str, pred_str = str(s.gt_dsm_path), str(pred_path)
        mae          = misc.align_dsm_with_gt(gt_str, pred_str, filter_water=False, filter_foliage=False)
        mae_no_water = misc.align_dsm_with_gt(gt_str, pred_str, filter_water=True,  filter_foliage=False)
        mae_f, aligned_dsm_f, gt_dsm_f = misc.align_dsm_with_gt(
            gt_str, pred_str, filter_water=True, filter_foliage=True, return_arrays=True
        )

        print(
            f"  [{s.pair_id[:55]:55s}]"
            f"  raw={mae:.4f}  no-water={mae_no_water:.4f}  filtered={mae_f:.4f} m"
        )

        gt_path = pair_dir / "gt_dsm.tif"
        with rasterio.open(
            str(gt_path), "w", driver="GTiff",
            height=gt_dsm_np.shape[0], width=gt_dsm_np.shape[1],
            count=1, dtype="float32", crs=dsm_meta["crs"],
            transform=dsm_meta["transform"], nodata=dsm_meta.get("nodata", np.nan),
        ) as dst:
            dst.write(gt_dsm_np.astype(np.float32), 1)

        # Save MAE filtered error map as float32 GeoTIFF
        error_map = np.abs(aligned_dsm_f - gt_dsm_f).astype(np.float32)
        error_path = pair_dir / "error_map_filtered.tif"
        with rasterio.open(
            str(error_path), "w", driver="GTiff",
            height=error_map.shape[0], width=error_map.shape[1],
            count=1, dtype="float32", crs=pred_meta["crs"],
            transform=pred_meta["transform"], nodata=pred_meta.get("nodata", np.nan),
        ) as dst:
            dst.write(error_map, 1)


        # Save comparison PNG: GT | Pred | Error
        # _save_comparison_png(pair_dir, gt_dsm_np, pred_dsm_np, mae_f, s.pair_id)

        row.update(dict(
            pred_dsm_path = str(pred_path),
            mae           = f"{mae:.6f}",
            mae_filtered  = f"{mae_f:.6f}",
            mae_no_water  = f"{mae_no_water:.6f}",
            status        = "ok",
            error         = "",
        ))

    except Exception as e:
        row.update(status="error", error=str(e))
        (pair_dir / "triangulation_error.txt").write_text(str(e) + "\n")
        print(f"  [FAIL] {s.pair_id[:55]}  {e}")

    return row


def _save_comparison_png(pair_dir: Path, gt: np.ndarray,
                         pred: np.ndarray, mae_f: float, title: str):
    finite_gt   = gt[np.isfinite(gt)]
    alt_min     = float(np.percentile(finite_gt, 2))  if finite_gt.size else 0
    alt_max     = float(np.percentile(finite_gt, 98)) if finite_gt.size else 1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(gt,   cmap="terrain", vmin=alt_min, vmax=alt_max)
    axes[0].set_title("GT DSM")
    axes[1].imshow(pred, cmap="terrain", vmin=alt_min, vmax=alt_max)
    axes[1].set_title("Predicted DSM")
    im = axes[2].imshow(np.abs(pred - gt), cmap="hot", vmin=0, vmax=10)
    plt.colorbar(im, ax=axes[2], label="Error (m)")
    axes[2].set_title(f"|Pred−GT|  MAE_filtered={mae_f:.3f} m")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title[:80], fontsize=8)
    plt.tight_layout()
    fig.savefig(str(pair_dir / "comparison.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_batches(samples: List[PairSample], predictor, model_name: str,
                output_dir: Path, batch_size: int = 1,
                max_side: int = 1024, border_trim_k: int = 32,
                skip_existing: bool = True):

    _init_summary(output_dir)
    done_pairs = _load_done_pairs(output_dir) if skip_existing else set()

    # Filter samples that are already done
    pending = [s for s in samples
               if (model_name, s.pair_id) not in done_pairs]

    if skip_existing and len(pending) < len(samples):
        print(f"  Skipping {len(samples) - len(pending)} already-computed pairs.")

    if not pending:
        print("  Nothing to do.")
        return

    device = predictor.device
    for i in tqdm(range(0, len(pending), batch_size),
                  total=(len(pending) + batch_size - 1) // batch_size,
                  desc=model_name):
        chunk = pending[i: i + batch_size]
        batch_L, batch_R, metas = _collate_minibatch(
            chunk, max_side=max_side, device=device
        )
        disp_np = predictor.predict_batch(batch_L, batch_R)  # (B, H, W)

        for bi, meta in enumerate(metas):
            lpad, rpad, tpad, bpad = meta["pad"]
            H, W = batch_L.shape[-2:]
            L = batch_L[bi, :, tpad: H-bpad, lpad: W-rpad]
            R = batch_R[bi, :, tpad: H-bpad, lpad: W-rpad]
            d_full = disp_np[bi]
            d = d_full[tpad: d_full.shape[0]-bpad, lpad: d_full.shape[1]-rpad]

            _ensure_dir(meta["pair"].pair_outdir())
            row = _save_pair_outputs_and_row(
                meta=meta, disp=d,
                imgL_3hw=L, imgR_3hw=R,
                border_trim_k=border_trim_k,
                model_name=model_name,
            )
            _append_summary(output_dir, row)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery functions
# ─────────────────────────────────────────────────────────────────────────────

def discover_omaha(test_root: Path, output_dir: Path,
                   sync: bool = True) -> List[PairSample]:
    """
    OMA layout (flat):
      {test_root}/OMA/test_omaha_{sync|diachronic}/{sync|diachronic}/{L,R,homography}
      RPC:  {test_root}/OMA/all_oma_and_jax_dsm_and_rpc/root_dir/{stem}.json
      DSM:  {test_root}/OMA/all_oma_and_jax_dsm_and_rpc/Track3-Truth/{AOI}_DSM.tif
    """
    subset    = "synchronic" if sync else "diachronic"
    sub_dir   = "synchronic" if sync else "diachronic"   # note: original has typo
    dataset   = f"omaha_{subset}"

    stereo_dir = test_root / "OMA" / f"test_omaha_{subset}" / sub_dir
    L_dir  = stereo_dir / "L"
    R_dir  = stereo_dir / "R"
    H_dir  = stereo_dir / "homography"
    rpc_dir = test_root / OMA_JAX_RPC_REL
    dsm_dir = test_root / OMA_JAX_DSM_REL

    if not L_dir.exists():
        print(f"[WARN] {L_dir} not found — skipping {dataset}")
        return []

    samples = []
    for lp in sorted(L_dir.glob("*.iio")):
        stem = lp.stem
        if "-" not in stem:
            continue

        left_id, right_id = stem.split("-", 1)
        aoi = "_".join(left_id.split("_")[:2])

        rp  = R_dir / f"{stem}.iio"
        hp  = H_dir / f"{stem}.npz"
        lrpc = rpc_dir / f"{left_id}.json"
        rrpc = rpc_dir / f"{right_id}.json"
        dsm  = dsm_dir / f"{aoi}_DSM.tif"

        if not rp.exists():
            continue
        if not lrpc.exists() or not rrpc.exists():
            continue
        if not dsm.exists():
            continue

        samples.append(PairSample(
            dataset         = dataset,
            pair_id         = stem,
            aoi             = aoi,
            left_path       = lp,
            right_path      = rp,
            out_dir         = output_dir,
            homography_path = hp if hp.exists() else None,
            left_rpc_json   = lrpc,
            right_rpc_json  = rrpc,
            gt_dsm_path     = dsm,
        ))

    print(f"  {dataset}: {len(samples)} pairs")
    return samples


def discover_jax(test_root: Path, output_dir: Path) -> List[PairSample]:
    """
    JAX layout (flat stereo pairs, per-AOI RPCs):
      {test_root}/JAX/test_jax/stereo_pairs_ba/{L,R,homography}
      RPC:  JAX_RPC_DIR/{JAX_004}/{stem}.json
      DSM:  JAX_DSM_DIR/{AOI}_DSM.tif
    """
    stereo_dir = test_root / "JAX" / "test_jax" / "stereo_pairs_ba"
    L_dir = stereo_dir / "L"
    R_dir = stereo_dir / "R"
    H_dir = stereo_dir / "homography"
    rpc_root = test_root / JAX_RPC_REL
    dsm_dir = test_root / JAX_DSM_REL

    if not L_dir.exists():
        print(f"[WARN] {L_dir} not found — skipping jax")
        return []

    samples = []
    for lp in sorted(L_dir.glob("*.iio")):
        stem = lp.stem
        if "-" not in stem:
            continue

        left_id, right_id = stem.split("-", 1)
        aoi = "_".join(left_id.split("_")[:2])   # JAX_004

        rp   = R_dir / f"{stem}.iio"
        hp   = H_dir / f"{stem}.npz"
        lrpc = rpc_root / aoi / f"{left_id}.json"
        rrpc = rpc_root / aoi / f"{right_id}.json"
        dsm  = dsm_dir / f"{aoi}_DSM.tif"

        if not rp.exists():
            continue
        if not lrpc.exists() or not rrpc.exists():
            continue
        if not dsm.exists():
            continue

        samples.append(PairSample(
            dataset         = "jax",
            pair_id         = stem,
            aoi             = aoi,
            left_path       = lp,
            right_path      = rp,
            out_dir         = output_dir,
            homography_path = hp if hp.exists() else None,
            left_rpc_json   = lrpc,
            right_rpc_json  = rrpc,
            gt_dsm_path     = dsm,
        ))

    print(f"  jax: {len(samples)} pairs")
    return samples


def discover_buenos_aires(test_root: Path, output_dir: Path) -> List[PairSample]:
    """
    Buenos Aires IARPA layout (per-AOI subfolders, long WV filenames):
      {test_root}/IARPA/test_buenos_aires/stereo_pairs_ba/{IARPA_001}/{L,R,homography}
      RPC:  BA_RPC_DIR/{IARPA_001}/{stem}.json
      DSM:  BA_DSM_DIR/{IARPA_001}_DSM.tif

    pair_id = '{IARPA_001}/{stem}' so outputs are grouped by AOI.
    """
    stereo_root = test_root / "IARPA" / "test_buenos_aires" / "stereo_pairs_ba"
    rpc_root = test_root / BA_RPC_REL
    dsm_dir = test_root / BA_DSM_REL

    if not stereo_root.exists():
        print(f"[WARN] {stereo_root} not found — skipping buenos_aires")
        return []

    samples = []
    for aoi_dir in sorted(p for p in stereo_root.iterdir() if p.is_dir()):
        aoi   = aoi_dir.name   # IARPA_001, IARPA_002, IARPA_003
        L_dir = aoi_dir / "L"
        R_dir = aoi_dir / "R"
        H_dir = aoi_dir / "homography"

        if not (L_dir.exists() and R_dir.exists()):
            continue

        rpc_dir = rpc_root / aoi
        dsm     = dsm_dir / f"{aoi}_DSM.tif"

        # Build set of valid RPC stems for reliable split of long WV filenames
        valid_tokens = {p.stem for p in rpc_dir.glob("*.json")} \
                       if rpc_dir.exists() else set()

        for lp in sorted(L_dir.glob("*.iio")):
            stem = lp.stem   # may contain multiple hyphens (WV satellite IDs)

            rp = R_dir / f"{stem}.iio"
            hp = H_dir / f"{stem}.npz"

            if not rp.exists():
                continue

            # Split stem into left/right tokens robustly
            split = _split_wv_pair(stem, valid_tokens)
            if split is None:
                continue
            left_tok, right_tok = split

            lrpc = rpc_dir / f"{left_tok}.json"
            rrpc = rpc_dir / f"{right_tok}.json"

            if not (lrpc.exists() and rrpc.exists()):
                continue
            if not dsm.exists():
                continue

            pair_id = f"{aoi}/{stem}"
            samples.append(PairSample(
                dataset         = "buenos_aires",
                pair_id         = pair_id,
                aoi             = aoi,
                left_path       = lp,
                right_path      = rp,
                out_dir         = output_dir,
                homography_path = hp if hp.exists() else None,
                left_rpc_json   = lrpc,
                right_rpc_json  = rrpc,
                gt_dsm_path     = dsm,
            ))

    print(f"  buenos_aires: {len(samples)} pairs")
    return samples


def _split_wv_pair(stem: str, valid_tokens: set) -> Optional[Tuple[str, str]]:
    """Split a WV-format pair stem into (left_token, right_token).
    Tries every hyphen position; prefers splits where both tokens are in valid_tokens.
    Falls back to the regex pattern used in the original script."""
    for i, ch in enumerate(stem):
        if ch == "-":
            left, right = stem[:i], stem[i+1:]
            if left in valid_tokens and right in valid_tokens:
                return left, right
    # Fallback: hyphen before a WV date pattern
    m = re.search(r"-(?=\d{2}[A-Z]{3}\d{2}WV0\d)", stem)
    if m:
        return stem[:m.start()], stem[m.end():]
    return None


def set_reproducible(seed: int = 42):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CHOICES = ["omaha_sync", "omaha_diach", "jax", "buenos_aires", "all"]


def main():
    set_reproducible(42)

    p = argparse.ArgumentParser(
        description="Evaluate MonSter++ checkpoints on test sets — altitude MAE."
    )
    p.add_argument("--models", nargs="+", required=True,
                   metavar="NAME:CKPT_PATH",
                   help="Models to evaluate, e.g. baseline:/path/to/final.pth")
    p.add_argument("--test-root", type=Path, required=True,
                   help="Root of test data, e.g. .../diachronic-stereo-synthetic/test")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Root output dir for all results")
    p.add_argument("--depth-anything", type=str, required=True,
                   help="Path to the Depth-Anything V2 checkpoint used by MonSter")
    p.add_argument("--datasets", nargs="+", default=["all"],
                   choices=DATASET_CHOICES,
                   help="Which test sets to run (default: all)")
    p.add_argument("--device",       default="cuda:0")
    p.add_argument("--batch-size",   type=int,  default=1)
    p.add_argument("--max-side",     type=int,  default=1024)
    p.add_argument("--border-trim",  type=int,  default=32)
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip (model, pair_id) rows already marked ok in summary.csv")
    p.add_argument("--no-skip-existing", dest="skip_existing",
                   action="store_false",
                   help="Re-run everything even if already computed")
    p.add_argument("--limit", type=int, default=0,
               help="Max pairs to evaluate per model (0 = all). For debugging.")
    args = p.parse_args()

    # ── Parse models ──────────────────────────────────────────────────────────
    models = {}
    for spec in args.models:
        if ":" not in spec:
            p.error(f"Invalid model spec '{spec}'. Expected name:ckpt_path")
        name, ckpt = spec.split(":", 1)
        if not Path(ckpt).exists():
            p.error(f"Checkpoint not found: {ckpt}")
        models[name] = ckpt

    print(f"Models ({len(models)}):")
    for name, ckpt in models.items():
        print(f"  {name:30s} {ckpt}")

    # ── Resolve datasets ──────────────────────────────────────────────────────
    run_all = "all" in args.datasets
    run = {
        "omaha_sync":    run_all or "omaha_sync"    in args.datasets,
        "omaha_diach":   run_all or "omaha_diach"   in args.datasets,
        "jax":           run_all or "jax"           in args.datasets,
        "buenos_aires":  run_all or "buenos_aires"  in args.datasets,
    }

    # ── Discover pairs for each dataset (once, shared across models) ──────────
    print("\nDiscovering pairs...")
    all_samples: List[PairSample] = []

    # Temporarily set output_dir to a placeholder 
    dummy = args.output_dir

    if run["omaha_diach"]:
        all_samples += discover_omaha(args.test_root, dummy, sync=False)
    if run["omaha_sync"]:
        all_samples += discover_omaha(args.test_root, dummy, sync=True)
    if run["jax"]:
        all_samples += discover_jax(args.test_root, dummy)
    if run["buenos_aires"]:
        all_samples += discover_buenos_aires(args.test_root, dummy)
    
    print(f"\nTotal pairs: {len(all_samples)}")
    if not all_samples:
        print("Nothing to evaluate.")
        return 
    
    if args.limit and args.limit > 0:
        all_samples = all_samples[:args.limit]
        print(f"  [DEBUG] Limited to {args.limit} pairs")

    # ── Init shared summary CSV ───────────────────────────────────────────────
    _ensure_dir(args.output_dir)
    _init_summary(args.output_dir)

    # ── Evaluate each model ───────────────────────────────────────────────────
    for model_name, ckpt_path in models.items():
        print(f"\n{'═'*65}")
        print(f"  Model: {model_name}")
        print(f"  Ckpt:  {ckpt_path}")
        print(f"{'═'*65}")

        # Each model gets its own subdirectory; pair paths are relative to it
        model_out = args.output_dir / model_name
        _ensure_dir(model_out)

        # Re-bind out_dir in each sample to this model's directory
        model_samples = [
            PairSample(
                dataset         = s.dataset,
                pair_id         = s.pair_id,
                aoi             = s.aoi,
                left_path       = s.left_path,
                right_path      = s.right_path,
                out_dir         = model_out,
                homography_path = s.homography_path,
                left_rpc_json   = s.left_rpc_json,
                right_rpc_json  = s.right_rpc_json,
                gt_dsm_path     = s.gt_dsm_path,
            )
            for s in all_samples
        ]

        predictor = MonSterPredictor(
            ckpt                   = ckpt_path,
            depth_anything_v2_path = args.depth_anything,
            device                 = args.device,
        )

        run_batches(
            samples        = model_samples,
            predictor      = predictor,
            model_name     = model_name,
            output_dir     = args.output_dir,   # shared summary.csv
            batch_size     = args.batch_size,
            max_side       = args.max_side,
            border_trim_k  = args.border_trim,
            skip_existing  = args.skip_existing,
        )

        del predictor
        torch.cuda.empty_cache()

        # ── Per-model aggregate from summary.csv ──────────────────────────────
        _print_model_summary(args.output_dir, model_name)

    # ── Final cross-model table ───────────────────────────────────────────────
    _print_comparison_table(args.output_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Summary printing
# ─────────────────────────────────────────────────────────────────────────────

def _print_model_summary(output_dir: Path, model_name: str):
    rows = []
    p = _summary_path(output_dir)
    if not p.exists():
        return
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row["model"] == model_name and row["status"] == "ok":
                rows.append(row)
    if not rows:
        return

    from collections import defaultdict
    by_dataset: dict = defaultdict(list)
    for r in rows:
        by_dataset[r["dataset"]].append(float(r["mae_filtered"]))

    print(f"\n  Summary — {model_name}")
    print(f"  {'dataset':20s}  {'MAE filtered (m)':>18}  {'n':>5}")
    print(f"  {'─'*20}  {'─'*18}  {'─'*5}")
    for ds, vals in sorted(by_dataset.items()):
        print(f"  {ds:20s}  {np.mean(vals):18.4f}  {len(vals):>5}")


def _print_comparison_table(output_dir: Path):
    from collections import defaultdict
    data: dict = defaultdict(lambda: defaultdict(list))
    p = _summary_path(output_dir)
    if not p.exists():
        return
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                data[row["model"]][row["dataset"]].append(
                    float(row["mae_filtered"])
                )

    if not data:
        return

    datasets = sorted({ds for m in data.values() for ds in m})
    models   = list(data.keys())

    print(f"\n{'═'*80}")
    print(f"  FINAL COMPARISON — MAE filtered (m)")
    print(f"{'═'*80}")
    header = f"  {'model':30s}" + "".join(f"  {ds[:14]:>14s}" for ds in datasets)
    print(header)
    print(f"  {'─'*30}" + "  " + "  ".join("─"*14 for _ in datasets))
    for m in models:
        row_str = f"  {m:30s}"
        for ds in datasets:
            vals = data[m].get(ds, [])
            row_str += f"  {np.mean(vals):14.4f}" if vals else f"  {'—':>14s}"
        print(row_str)
    print(f"{'═'*80}")
    print(f"\n  Full results: {_summary_path(output_dir)}")


if __name__ == "__main__":
    main()
