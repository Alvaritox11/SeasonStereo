import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import iio
import numpy as np
import torch
import torch.nn.functional as F
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim


# ---------------------------
# IO
# ---------------------------

def load_image(path: Path) -> np.ndarray:
    img = iio.read(str(path))
    img = np.asarray(img)

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)

    if img.ndim != 3:
        raise ValueError(f"Expected image with 2 or 3 dims, got shape={img.shape} for {path}")

    if img.shape[2] > 3:
        img = img[..., :3]

    return img.astype(np.float32)


def load_disp(path: Path) -> np.ndarray:
    disp = iio.read(str(path))
    disp = np.asarray(disp)

    if disp.ndim == 3:
        disp = disp[..., 0]

    if disp.ndim != 2:
        raise ValueError(f"Expected disparity with 2 dims, got shape={disp.shape} for {path}")

    return disp.astype(np.float32)


# ---------------------------
# Tensor helpers
# ---------------------------

def rgb_image_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)


def single_channel_to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


# ---------------------------
# Warping
# ---------------------------

def warp_right_to_left(
    right_img: torch.Tensor,
    disp_left: torch.Tensor,
    padding_mode: str = "zeros",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp right image into the left view using left-referenced disparity.
    """
    assert right_img.dim() == 4 and disp_left.dim() == 4
    B, C, H, W = right_img.shape
    assert disp_left.shape == (B, 1, H, W)

    y, x = torch.meshgrid(
        torch.arange(H, device=right_img.device, dtype=right_img.dtype),
        torch.arange(W, device=right_img.device, dtype=right_img.dtype),
        indexing="ij",
    )
    x = x[None, None, ...].expand(B, -1, -1, -1)
    y = y[None, None, ...].expand(B, -1, -1, -1)

    x_right = x - disp_left
    y_right = y

    valid = (x_right >= 0) & (x_right <= (W - 1)) & (y_right >= 0) & (y_right <= (H - 1))
    valid = valid.to(right_img.dtype)

    if W == 1:
        x_norm = torch.zeros_like(x_right)
    else:
        x_norm = 2.0 * (x_right / (W - 1)) - 1.0

    if H == 1:
        y_norm = torch.zeros_like(y_right)
    else:
        y_norm = 2.0 * (y_right / (H - 1)) - 1.0

    grid = torch.cat([x_norm, y_norm], dim=1)
    grid = grid.permute(0, 2, 3, 1).contiguous()

    warped = F.grid_sample(
        right_img,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped, valid


# ---------------------------
# Masks
# ---------------------------

def disparity_valid_mask(disp: np.ndarray, require_positive: bool = False) -> np.ndarray:
    mask = np.isfinite(disp)
    if require_positive:
        mask &= disp > 0
    return mask


def build_final_mask(
    warp_disp: np.ndarray,
    warp_valid: np.ndarray,
    gt_disp: Optional[np.ndarray] = None,
    require_positive_warp: bool = False,
    require_positive_gt: bool = False,
) -> np.ndarray:
    mask = disparity_valid_mask(warp_disp, require_positive=require_positive_warp)
    mask &= warp_valid.astype(bool)

    if gt_disp is not None:
        mask &= disparity_valid_mask(gt_disp, require_positive=require_positive_gt)

    return mask


# ---------------------------
# EPE
# ---------------------------

def compute_epe(gt_disp: np.ndarray, pred_disp: np.ndarray, require_positive_gt: bool = False) -> float:
    mask = disparity_valid_mask(gt_disp, require_positive=require_positive_gt)
    if mask.sum() == 0:
        return float("nan")
    err = np.abs(gt_disp - pred_disp)
    return float(err[mask].mean())


# ---------------------------
# Lab helpers
# ---------------------------

def rgb_to_lab_pair(left: np.ndarray, left_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left_lab = rgb2lab(np.clip(left / 255.0, 0.0, 1.0)).astype(np.float32)
    left_hat_lab = rgb2lab(np.clip(left_hat / 255.0, 0.0, 1.0)).astype(np.float32)
    return left_lab, left_hat_lab


# ---------------------------
# SSIM helpers
# ---------------------------

def reduce_metric_map(diff_map: np.ndarray, valid_mask: np.ndarray, channel_axis: Optional[int]) -> float:
    valid_mask = valid_mask.astype(bool)

    if channel_axis is None:
        if valid_mask.sum() == 0:
            return 0.0
        return float(diff_map[valid_mask].mean())

    if diff_map.ndim != 3:
        raise ValueError(f"Expected diff_map with 3 dims when channel_axis is set, got {diff_map.shape}")

    if channel_axis == 0:
        valid = np.broadcast_to(valid_mask[None, ...], diff_map.shape)
    elif channel_axis == -1 or channel_axis == 2:
        valid = np.broadcast_to(valid_mask[..., None], diff_map.shape)
    else:
        raise ValueError(f"Unsupported channel_axis={channel_axis}")

    if valid.sum() == 0:
        return 0.0

    return float(diff_map[valid].mean())


def compute_ssim_rgb(left: np.ndarray, left_hat: np.ndarray, valid_mask: np.ndarray) -> float:
    _, diff = ssim(
        left,
        left_hat,
        full=True,
        channel_axis=-1,
        data_range=255.0,
    )
    return reduce_metric_map(diff, valid_mask, channel_axis=-1)


def compute_ssim_lab(left: np.ndarray, left_hat: np.ndarray, valid_mask: np.ndarray) -> float:
    left_lab, left_hat_lab = rgb_to_lab_pair(left, left_hat)
    _, diff = ssim(
        left_lab,
        left_hat_lab,
        full=True,
        channel_axis=-1,
        data_range=255.0,
    )
    return reduce_metric_map(diff, valid_mask, channel_axis=-1)


def compute_ssim_ab(left: np.ndarray, left_hat: np.ndarray, valid_mask: np.ndarray) -> float:
    left_lab, left_hat_lab = rgb_to_lab_pair(left, left_hat)
    left_ab = left_lab[..., 1:3]
    left_hat_ab = left_hat_lab[..., 1:3]
    _, diff = ssim(
        left_ab,
        left_hat_ab,
        full=True,
        channel_axis=-1,
        data_range=255.0,
    )
    return reduce_metric_map(diff, valid_mask, channel_axis=-1)


def compute_color_distance_stats(
    left: np.ndarray,
    left_hat: np.ndarray,
    valid_mask: np.ndarray,
    mode: str = "ab_l2",
) -> Dict[str, float]:
    left_lab, left_hat_lab = rgb_to_lab_pair(left, left_hat)

    if mode == "ab_l2":
        delta = np.linalg.norm(left_lab[..., 1:3] - left_hat_lab[..., 1:3], axis=-1)
        prefix = "ab"
    elif mode == "lab_l2":
        delta = np.linalg.norm(left_lab - left_hat_lab, axis=-1)
        prefix = "lab"
    elif mode == "ciede2000":
        delta = deltaE_ciede2000(left_lab, left_hat_lab)
        prefix = "ciede2000"
    elif mode == "rgb_l2":
        delta = np.linalg.norm(left - left_hat, axis=-1)
        prefix = "rgb"
    else:
        raise ValueError(f"Unsupported color_distance_mode={mode}")

    vals = delta[valid_mask]
    if vals.size == 0:
        return {
            f"mean_color_distance_{prefix}": float("nan"),
            f"median_color_distance_{prefix}": float("nan"),
            f"p95_color_distance_{prefix}": float("nan"),
            f"color_distance_{prefix}": float("nan"),
        }

    mean_v = float(vals.mean())
    median_v = float(np.median(vals))
    p95_v = float(np.percentile(vals, 95))
    color_distance = float(0.5 * mean_v + 0.5 * p95_v)

    return {
        f"mean_color_distance_{prefix}": mean_v,
        f"median_color_distance_{prefix}": median_v,
        f"p95_color_distance_{prefix}": p95_v,
        f"color_distance_{prefix}": color_distance,
    }


# ---------------------------
# DINOv3
# ---------------------------

def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class DinoV3Metric:
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vitl16-pretrain-sat493m",
        device: str = "cuda",
        repo_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        arch: str = "dinov3_vitl16",
    ):
        self.device = device
        self.repo_path = Path(repo_path) if repo_path else None
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self.arch = arch

        if self.repo_path is not None:
            self._init_from_local_repo()
        else:
            self._init_from_huggingface(model_name)

    def _init_from_huggingface(self, model_name: str) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.mode = "huggingface"

    def _init_from_local_repo(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(f"DINOv3 repo path does not exist: {self.repo_path}")

        hubconf_path = self.repo_path / "hubconf.py"
        if not hubconf_path.exists():
            raise FileNotFoundError(f"Could not find hubconf.py in repo path: {self.repo_path}")

        hubconf = _load_module_from_file("_dinov3_hubconf_local", hubconf_path)

        if not hasattr(hubconf, self.arch):
            raise AttributeError(
                f"Could not find constructor '{self.arch}' in {hubconf_path}. "
                f"Check the official repo function name."
            )

        ctor = getattr(hubconf, self.arch)
        try:
            self.model = ctor(pretrained=False)
        except TypeError:
            self.model = ctor()

        if self.ckpt_path is not None:
            if not self.ckpt_path.exists():
                raise FileNotFoundError(f"DINOv3 checkpoint not found: {self.ckpt_path}")
            state = torch.load(str(self.ckpt_path), map_location="cpu")
            if isinstance(state, dict):
                for key in ["state_dict", "model", "teacher", "student"]:
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            self.missing_keys = list(missing)
            self.unexpected_keys = list(unexpected)
        else:
            self.missing_keys = []
            self.unexpected_keys = []

        self.model = self.model.to(self.device)
        self.model.eval()
        self.mode = "local_repo"

    def compute(self, left: np.ndarray, left_hat: np.ndarray, valid_mask: np.ndarray) -> float:
        left_masked = left.copy()
        left_hat_masked = left_hat.copy()

        left_masked[~valid_mask] = 0
        left_hat_masked[~valid_mask] = 0

        if self.mode == "huggingface":
            inputs = self.processor(
                images=[left_masked.astype(np.uint8), left_hat_masked.astype(np.uint8)],
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)

            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                f1 = outputs.pooler_output[0]
                f2 = outputs.pooler_output[1]
            else:
                f1 = outputs.last_hidden_state[0].mean(dim=0)
                f2 = outputs.last_hidden_state[1].mean(dim=0)
        else:
            imgs = [left_masked.astype(np.uint8), left_hat_masked.astype(np.uint8)]
            tensors = []
            for img in imgs:
                x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensors.append(x)
            batch = torch.stack(tensors, dim=0).to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)

            if isinstance(outputs, dict):
                if "x_norm_clstoken" in outputs:
                    feats = outputs["x_norm_clstoken"]
                elif "x_prenorm" in outputs:
                    feats = outputs["x_prenorm"].mean(dim=1)
                elif "last_hidden_state" in outputs:
                    feats = outputs["last_hidden_state"].mean(dim=1)
                else:
                    first_val = next(iter(outputs.values()))
                    feats = first_val.mean(dim=1) if first_val.ndim == 3 else first_val
            else:
                feats = outputs.mean(dim=1) if outputs.ndim == 3 else outputs

            f1, f2 = feats[0], feats[1]

        sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0), dim=-1)
        return float(sim.item())


# ---------------------------
# Shared warped inputs
# ---------------------------

def prepare_warped_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    warp_disp: np.ndarray,
    gt_disp: Optional[np.ndarray] = None,
    require_positive_warp: bool = False,
    require_positive_gt: bool = False,
) -> Dict[str, np.ndarray]:
    left_t = rgb_image_to_tensor(left_img)
    right_t = rgb_image_to_tensor(right_img)

    warp_disp_clean = np.nan_to_num(warp_disp, nan=0.0, posinf=0.0, neginf=0.0)
    warp_disp_t = single_channel_to_tensor(warp_disp_clean)

    left_hat_t, warp_valid_t = warp_right_to_left(right_t, warp_disp_t)

    left_hat = left_hat_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    warp_valid = warp_valid_t.squeeze().cpu().numpy().astype(bool)

    final_mask = build_final_mask(
        warp_disp=warp_disp,
        warp_valid=warp_valid,
        gt_disp=gt_disp,
        require_positive_warp=require_positive_warp,
        require_positive_gt=require_positive_gt,
    )

    return {
        "left": left_img.astype(np.float32),
        "left_hat": left_hat.astype(np.float32),
        "warp_valid": warp_valid,
        "final_mask": final_mask,
    }