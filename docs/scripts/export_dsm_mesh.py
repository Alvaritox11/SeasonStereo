#!/usr/bin/env python3
"""Export a DSM GeoTIFF as a lightweight untextured GLB mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import trimesh


RAMP = np.array(
    [
        [24, 48, 79, 255],
        [37, 120, 166, 255],
        [85, 192, 160, 255],
        [224, 194, 94, 255],
        [185, 91, 63, 255],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input DSM GeoTIFF.")
    parser.add_argument("--output", required=True, type=Path, help="Output GLB path.")
    parser.add_argument("--stride", type=int, default=3, help="Pixel stride for mesh decimation.")
    parser.add_argument(
        "--z-offset",
        choices=("min", "median", "mean", "zero"),
        default="min",
        help="Height reference subtracted from all DSM values.",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=1.0,
        help="Multiplier applied after subtracting the height reference.",
    )
    return parser.parse_args()


def make_colors(heights: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(heights, [2, 98])
    denom = max(float(high - low), 1e-6)
    t = np.clip((heights - low) / denom, 0.0, 1.0)
    scaled = t * (len(RAMP) - 1)
    idx = np.minimum(np.floor(scaled).astype(np.int32), len(RAMP) - 2)
    local = (scaled - idx)[:, None]
    colors = RAMP[idx] * (1.0 - local) + RAMP[idx + 1] * local
    return np.clip(colors, 0, 255).astype(np.uint8)


def height_reference(values: np.ndarray, mode: str) -> float:
    if mode == "min":
        return float(np.nanmin(values))
    if mode == "median":
        return float(np.nanmedian(values))
    if mode == "mean":
        return float(np.nanmean(values))
    return 0.0


def export_mesh(input_path: Path, output_path: Path, stride: int, z_offset: str, vertical_scale: float) -> dict:
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    with rasterio.open(input_path) as dataset:
        dsm = dataset.read(1).astype(np.float32)
        transform = dataset.transform
        crs = str(dataset.crs) if dataset.crs else None

    dsm = dsm[::stride, ::stride]
    rows, cols = dsm.shape
    valid = np.isfinite(dsm)
    valid_values = dsm[valid]
    if valid_values.size == 0:
        raise ValueError(f"No finite DSM values found in {input_path}")

    ref = height_reference(valid_values, z_offset)
    row_grid, col_grid = np.mgrid[0:rows, 0:cols]
    src_rows = row_grid * stride + 0.5
    src_cols = col_grid * stride + 0.5

    xs = transform.c + transform.a * src_cols + transform.b * src_rows
    ys = transform.f + transform.d * src_cols + transform.e * src_rows

    if transform.is_identity:
        xs = src_cols
        ys = src_rows

    center_x = float(np.nanmean(xs[valid]))
    center_y = float(np.nanmean(ys[valid]))

    vertex_index = -np.ones((rows, cols), dtype=np.int64)
    vertices = []
    heights = []

    for row in range(rows):
        for col in range(cols):
            if not valid[row, col]:
                continue
            vertex_index[row, col] = len(vertices)
            height = (float(dsm[row, col]) - ref) * vertical_scale
            vertices.append([float(xs[row, col] - center_x), height, float(-(ys[row, col] - center_y))])
            heights.append(float(dsm[row, col]))

    faces = []
    for row in range(rows - 1):
        for col in range(cols - 1):
            a = vertex_index[row, col]
            b = vertex_index[row, col + 1]
            c = vertex_index[row + 1, col]
            d = vertex_index[row + 1, col + 1]
            if min(a, b, c, d) < 0:
                continue
            faces.append([a, c, b])
            faces.append([b, c, d])

    vertices_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int64)
    heights_np = np.asarray(heights, dtype=np.float32)

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_colors=make_colors(heights_np),
        process=False,
    )
    mesh.metadata.update(
        {
            "source_dsm": input_path.name,
            "source_pair": input_path.parent.name,
            "crs": crs,
            "stride": stride,
            "height_reference": z_offset,
            "height_reference_value": ref,
            "vertical_scale": vertical_scale,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)

    return {
        "output": str(output_path),
        "vertices": int(len(vertices_np)),
        "faces": int(len(faces_np)),
        "height_min_m": float(np.nanmin(valid_values)),
        "height_max_m": float(np.nanmax(valid_values)),
        "height_reference_m": ref,
        "stride": stride,
    }


def main() -> None:
    args = parse_args()
    summary = export_mesh(args.input, args.output, args.stride, args.z_offset, args.vertical_scale)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
