#!/usr/bin/env python3
"""Export an IIO disparity map as a lightweight untextured GLB surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import iio
import numpy as np
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
    parser.add_argument("--input", required=True, type=Path, help="Input disparity .iio file.")
    parser.add_argument("--output", required=True, type=Path, help="Output GLB path.")
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for mesh decimation.")
    parser.add_argument(
        "--xy-scale",
        type=float,
        default=0.5,
        help="Horizontal scale applied to pixel coordinates.",
    )
    parser.add_argument(
        "--z-offset",
        choices=("min", "median", "mean", "zero"),
        default="min",
        help="Disparity reference subtracted from all values.",
    )
    parser.add_argument(
        "--vertical-scale",
        type=float,
        default=0.6,
        help="Multiplier applied after subtracting the disparity reference.",
    )
    return parser.parse_args()


def make_colors(values: np.ndarray) -> np.ndarray:
    low, high = np.nanpercentile(values, [2, 98])
    denom = max(float(high - low), 1e-6)
    t = np.clip((values - low) / denom, 0.0, 1.0)
    scaled = t * (len(RAMP) - 1)
    idx = np.minimum(np.floor(scaled).astype(np.int32), len(RAMP) - 2)
    local = (scaled - idx)[:, None]
    colors = RAMP[idx] * (1.0 - local) + RAMP[idx + 1] * local
    return np.clip(colors, 0, 255).astype(np.uint8)


def value_reference(values: np.ndarray, mode: str) -> float:
    if mode == "min":
        return float(np.nanmin(values))
    if mode == "median":
        return float(np.nanmedian(values))
    if mode == "mean":
        return float(np.nanmean(values))
    return 0.0


def read_disparity(input_path: Path) -> np.ndarray:
    disparity = np.asarray(iio.read(str(input_path)), dtype=np.float32)
    if disparity.ndim == 3:
        if disparity.shape[-1] != 1:
            raise ValueError(f"Expected a single-channel disparity map, got shape {disparity.shape}")
        disparity = disparity[..., 0]
    if disparity.ndim != 2:
        raise ValueError(f"Expected a 2D disparity map, got shape {disparity.shape}")
    return disparity


def export_mesh(
    input_path: Path,
    output_path: Path,
    stride: int,
    xy_scale: float,
    z_offset: str,
    vertical_scale: float,
) -> dict:
    if stride < 1:
        raise ValueError("--stride must be >= 1")

    disparity = read_disparity(input_path)
    disparity = disparity[::stride, ::stride]
    rows, cols = disparity.shape
    valid = np.isfinite(disparity)
    valid_values = disparity[valid]
    if valid_values.size == 0:
        raise ValueError(f"No finite disparity values found in {input_path}")

    ref = value_reference(valid_values, z_offset)
    center_col = (cols - 1) * 0.5
    center_row = (rows - 1) * 0.5

    vertex_index = -np.ones((rows, cols), dtype=np.int64)
    vertices = []
    values = []

    for row in range(rows):
        for col in range(cols):
            if not valid[row, col]:
                continue
            vertex_index[row, col] = len(vertices)
            value = float(disparity[row, col])
            x = (col - center_col) * stride * xy_scale
            z = (row - center_row) * stride * xy_scale
            y = (value - ref) * vertical_scale
            vertices.append([x, y, z])
            values.append(value)

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
    values_np = np.asarray(values, dtype=np.float32)

    mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np,
        vertex_colors=make_colors(values_np),
        process=False,
    )
    mesh.metadata.update(
        {
            "source_disparity": input_path.name,
            "source_pair": input_path.parent.name,
            "stride": stride,
            "xy_scale": xy_scale,
            "disparity_reference": z_offset,
            "disparity_reference_value": ref,
            "vertical_scale": vertical_scale,
            "units": "disparity pixels visualized as mesh height",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)

    return {
        "output": str(output_path),
        "vertices": int(len(vertices_np)),
        "faces": int(len(faces_np)),
        "disparity_min": float(np.nanmin(valid_values)),
        "disparity_max": float(np.nanmax(valid_values)),
        "disparity_reference": ref,
        "stride": stride,
        "xy_scale": xy_scale,
        "vertical_scale": vertical_scale,
    }


def main() -> None:
    args = parse_args()
    summary = export_mesh(
        args.input,
        args.output,
        args.stride,
        args.xy_scale,
        args.z_offset,
        args.vertical_scale,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
