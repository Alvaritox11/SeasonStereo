#!/usr/bin/env python3
import os
import time
import glob
import traceback

import numpy as np
import cv2
import iio  # pip package iio (the one you already use)
from skimage import transform
from scipy.ndimage import binary_erosion


# =========================
# CONFIG
# =========================

# 1) Base path for the reference rectified pairs and homographies
# Change this path if your reference data lives elsewhere.
BASE_DIR = "data/synchronic_only"

# 2) Paths for the original images (Source Dataset)
# Change this path if your Track3 data lives elsewhere.
TRACK3_ROOT = "data/Train-Track3-cropped"
TRACK3_OMA_DIR = os.path.join(TRACK3_ROOT, "Track3-RGB-2")  # OMA_xxx images
TRACK3_JAX_DIR = os.path.join(TRACK3_ROOT, "Track3-RGB-1")  # JAX_xxx images

# 3) Derived paths inside BASE_DIR
HOMOGRAPHY_DIR = os.path.join(BASE_DIR, "homography")
REF_L_DIR = os.path.join(BASE_DIR, "L")
REF_R_DIR = os.path.join(BASE_DIR, "R")

# 4) Log files
LOG_MISSING = os.path.join(BASE_DIR, "missing_files.txt")
LOG_COMPARISON = os.path.join(BASE_DIR, "comparison_errors.txt")
LOG_LOW_VALID = os.path.join(BASE_DIR, "low_valid_fraction.txt")

# 5) Geometry / thresholds
FULL_SHAPE = (2048, 2048)   # homographies were computed in this coord system
CROP_SIZE = 704             # your ROI used to derive output size
ERODE_PX = 3                # remove border fade region
# Thresholds in SAME SCALE as your images (you said 0..255 floats):
THR_MAX = 0.10              # recommended starting point (you saw ~0.0078)
THR_L1  = 0.002             # recommended starting point (you saw ~0.00019)
VALID_FRAC_MIN = 0.80

# Warp params
WARP_MODE = "constant"
WARP_CVAL = 0
WARP_ORDER = 1
PRESERVE_RANGE = True


# =========================
# CORE FUNCTIONS
# =========================

def warp_with_H(img, H, output_shape, flip=False, **warp_kwargs):
    H = np.asarray(H, dtype=np.float64)

    if flip:
        h, w = output_shape
        H_flip = np.array([[-1, 0, w - 1],
                           [0,  1, 0],
                           [0,  0, 1]], dtype=np.float64)
        H = H_flip @ H

    warped = transform.warp(
        img,
        transform.ProjectiveTransform(H).inverse,
        output_shape=output_shape,
        **warp_kwargs,
    )
    return warped, H

def bounding_box2D(pts):
    dim = len(pts[0])  # should be 2
    bb_min = [min([t[i] for t in pts]) for i in range(dim)]
    bb_max = [max([t[i] for t in pts]) for i in range(dim)]
    return bb_min[0], bb_min[1], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1]

def T(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]], dtype=np.float64)

def H_crop_to_dst_global(H_full, full_shape, crop_shape):
    """
    Option 1: keep H as if it were computed on the full image coords,
    but apply a translation so crop coords map into full coords first.
    H_crop = H_full @ T_add
    """
    H_full_h, H_full_w = full_shape
    crop_h, crop_w = crop_shape
    cx = (H_full_w - crop_w) / 2.0
    cy = (H_full_h - crop_h) / 2.0
    T_add = T(cx, cy)  # crop -> full
    return (H_full @ T_add), (cx, cy)

def valid_mask_from_warp(img_shape_hw, H, output_shape):
    """Mask=1 where warp samples real source pixels (no padding)."""
    mask = np.ones(img_shape_hw, dtype=np.float32)
    mask_w, _ = warp_with_H(
        mask, H, output_shape,
        preserve_range=True, mode="constant", cval=0, order=0
    )
    return mask_w > 0.5

def compare_masked(img_ref, img_test, valid, erode_px=0):
    """
    Compare in valid region only, optionally eroding valid to avoid fade borders.
    Returns l1_mean and max over valid region (err_map = mean abs diff over channels).
    """
    if img_ref.shape != img_test.shape:
        raise ValueError(f"shape mismatch: ref={img_ref.shape} test={img_test.shape}")

    v = valid.astype(bool)
    if erode_px > 0:
        se = np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=bool)
        v = binary_erosion(v, structure=se)

    if v.sum() == 0:
        return {"l1_mean": 0.0, "max": 0.0, "valid_fraction": 0.0}

    i1 = img_ref.astype(np.float32)
    i2 = img_test.astype(np.float32)
    diff = np.abs(i1 - i2)

    if diff.ndim == 3:
        err_map = diff.mean(axis=2)  # (H,W)
    else:
        err_map = diff

    l1 = float(err_map[v].mean())
    mx = float(err_map[v].max())
    frac = float(v.mean())

    return {"l1_mean": l1, "max": mx, "valid_fraction": frac}


# =========================
# PATH RESOLUTION
# =========================

def parse_pair_stem(stem):
    """
    stem example: "OMA_084_032_RGB-OMA_084_033_RGB"
    returns left_stem, right_stem
    """
    if "-" not in stem:
        raise ValueError(f"Unexpected pair stem (no '-'): {stem}")
    left, right = stem.split("-", 1)
    return left, right

def resolve_source_tif(stem, track3_oma_dir, track3_jax_dir):
    """
    stem example: "OMA_084_032_RGB" -> folder "OMA_084", file "OMA_084_032_RGB.tif"
    """
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected image stem: {stem}")

    region = parts[0]               # "OMA" or "JAX"
    tile = parts[1]                 # "084"
    folder = f"{region}_{tile}"

    root = track3_oma_dir if region == "OMA" else track3_jax_dir
    path = os.path.join(root, folder, f"{stem}.tif")

    # If .tif doesn't exist, try .tiff
    if not os.path.exists(path):
        path2 = os.path.join(root, folder, f"{stem}.tiff")
        if os.path.exists(path2):
            return path2
    return path


# =========================
# MAIN LOOP
# =========================

def run_dataset_check():
    # os.makedirs(BASE_DIR, exist_ok=True)

    # Reset logs each run
    with open(LOG_MISSING, "w") as f:
        f.write("")
    with open(LOG_COMPARISON, "w") as f:
        f.write("")
    with open(LOG_LOW_VALID, "w") as f:
        f.write("")

    refL_files = sorted(glob.glob(os.path.join(REF_L_DIR, "*.iio")))
    total = len(refL_files)

    n_ok = 0
    n_fail = 0
    n_missing = 0
    n_exception = 0

    def log_missing(msg):
        nonlocal n_missing
        n_missing += 1
        with open(LOG_MISSING, "a") as f:
            f.write(msg.rstrip() + "\n")

    def log_fail(msg):
        nonlocal n_fail
        n_fail += 1
        with open(LOG_COMPARISON, "a") as f:
            f.write(msg.rstrip() + "\n")

    def log_low_valid(msg):
        with open(LOG_LOW_VALID, "a") as f:
            f.write(msg.rstrip() + "\n")

    t0 = time.time()
    for idx, refL_path in enumerate(refL_files, start=1):
        fname = os.path.basename(refL_path)
        stem = os.path.splitext(fname)[0]

        refR_path = os.path.join(REF_R_DIR, fname)
        H_path = os.path.join(HOMOGRAPHY_DIR, f"{stem}.npz")

        # ---- check required files
        if not os.path.exists(refR_path):
            log_missing(f"[MISSING_REF_R] {stem} | expected: {refR_path}")
            continue
        if not os.path.exists(H_path):
            log_missing(f"[MISSING_H] {stem} | expected: {H_path}")
            continue

        try:
            # ---- load references
            refL = np.load(refL_path)
            refR = np.load(refR_path)

            # ---- parse stems for source images
            left_stem, right_stem = parse_pair_stem(stem)
            srcL_path = resolve_source_tif(left_stem, TRACK3_OMA_DIR, TRACK3_JAX_DIR)
            srcR_path = resolve_source_tif(right_stem, TRACK3_OMA_DIR, TRACK3_JAX_DIR)

            if not os.path.exists(srcL_path):
                log_missing(f"[MISSING_SRC_L] {stem} | expected: {srcL_path}")
                continue
            if not os.path.exists(srcR_path):
                log_missing(f"[MISSING_SRC_R] {stem} | expected: {srcR_path}")
                continue

            # ---- read sources
            imL = iio.read(srcL_path)
            imR = iio.read(srcR_path)

            # ---- load homographies (computed in FULL_SHAPE coords)
            b = np.load(H_path)
            Hleft_full = b["Hleft"]
            Hright_full = b["Hright"]

            # ---- Option: if sources are cropped (e.g. 1024), convert homographies
            # If sources are already 2048, keep as-is.
            if tuple(imL.shape[:2]) != FULL_SHAPE:
                Hleft, _ = H_crop_to_dst_global(Hleft_full, FULL_SHAPE, imL.shape[:2])
                Hright, _ = H_crop_to_dst_global(Hright_full, FULL_SHAPE, imR.shape[:2])
            else:
                Hleft = Hleft_full
                Hright = Hright_full

            # ---- compute output size from ROI (same logic you use)
            crop_size = CROP_SIZE
            x = imL.shape[1] // 2 - crop_size // 2
            y = imL.shape[0] // 2 - crop_size // 2
            w = h = crop_size

            if imL.shape[1] < crop_size or imL.shape[0] < crop_size:
                crop_size = 512
                x = imL.shape[1] // 2 - crop_size // 2
                y = imL.shape[0] // 2 - crop_size // 2
                w = h = crop_size

            roi = np.array([[x,     y],
                            [x + w, y],
                            [x + w, y + h],
                            [x,     y + h]], dtype=np.float32).reshape(-1, 1, 2)

            roi_after = cv2.perspectiveTransform(roi, Hleft)
            roi_flat = roi_after.reshape(-1, 2)
            x_float, y_float, w_float, h_float = bounding_box2D(roi_flat)
            x_bb, y_bb, w_bb, h_bb = np.round([x_float, y_float, w_float, h_float]).astype(int)

            # IMPORTANT: you only use (h_bb, w_bb) as output_shape (as in your snippet)
            out_shape = (h_bb, w_bb)

            # ---- warp sources
            warpedL, _ = warp_with_H(
                imL, Hleft, out_shape,
                preserve_range=PRESERVE_RANGE, mode=WARP_MODE, cval=WARP_CVAL, order=WARP_ORDER
            )
            warpedR, _ = warp_with_H(
                imR, Hright, out_shape,
                preserve_range=PRESERVE_RANGE, mode=WARP_MODE, cval=WARP_CVAL, order=WARP_ORDER
            )

            # ---- verify shapes
            if refL.shape[:2] != warpedL.shape[:2] or refL.shape != warpedL.shape:
                log_fail(f"[SHAPE_MISMATCH] {stem} | refL={refL.shape} warpedL={warpedL.shape} | out_shape={out_shape}")
                continue
            if refR.shape[:2] != warpedR.shape[:2] or refR.shape != warpedR.shape:
                log_fail(f"[SHAPE_MISMATCH] {stem} | refR={refR.shape} warpedR={warpedR.shape} | out_shape={out_shape}")
                continue

            # ---- valid mask from warp + erosion
            validL = valid_mask_from_warp(imL.shape[:2], Hleft, refL.shape[:2])
            validR = valid_mask_from_warp(imR.shape[:2], Hright, refR.shape[:2])

            mL = compare_masked(refL, warpedL, validL, erode_px=ERODE_PX)
            mR = compare_masked(refR, warpedR, validR, erode_px=ERODE_PX)

            if (mL["valid_fraction"] < VALID_FRAC_MIN) or (mR["valid_fraction"] < VALID_FRAC_MIN):
                log_low_valid(
                    f"[LOW_VALID] {stem} | "
                    f"L valid={mL['valid_fraction']:.4f} | "
                    f"R valid={mR['valid_fraction']:.4f} | "
                    f"src_shape={imL.shape[:2]} out_shape={out_shape}"
                )

            # ---- threshold check
            badL = (mL["max"] > THR_MAX) or (mL["l1_mean"] > THR_L1)
            badR = (mR["max"] > THR_MAX) or (mR["l1_mean"] > THR_L1)

            if badL or badR:
                log_fail(
                    f"[DIFF_TOO_HIGH] {stem} | "
                    f"L: l1={mL['l1_mean']:.6f} max={mL['max']:.6f} valid={mL['valid_fraction']:.4f} | "
                    f"R: l1={mR['l1_mean']:.6f} max={mR['max']:.6f} valid={mR['valid_fraction']:.4f} | "
                    f"src_shape={imL.shape[:2]} out_shape={out_shape}"
                )
            else:
                n_ok += 1

        except Exception as e:
            n_exception += 1
            log_fail(f"[EXCEPTION] {stem} | {repr(e)}\n{traceback.format_exc()}")

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - t0
            rate = idx / max(elapsed, 1e-6)
            remaining = (total - idx) / max(rate, 1e-6)
            print(f"[{idx}/{total}] ok={n_ok} fail={n_fail} missing={n_missing} exc={n_exception} | "
                f"elapsed={elapsed/60:.1f}m rate={rate:.2f}/s ETA={remaining/60:.1f}m")

    # Summary
    summary = (
        f"Total pairs (REF_L): {total}\n"
        f"OK:                 {n_ok}\n"
        f"Comparison fails:   {n_fail}\n"
        f"Missing files:      {n_missing}\n"
        f"Exceptions:         {n_exception}\n"
        f"Thresholds: max<={THR_MAX}, l1<={THR_L1}, erode_px={ERODE_PX}\n"
        f"Logs:\n  missing: {LOG_MISSING}\n  comparison: {LOG_COMPARISON}\n"
    )
    print(summary)
    with open(LOG_COMPARISON, "a") as f:
        f.write("\n" + summary)


if __name__ == "__main__":
    run_dataset_check()
