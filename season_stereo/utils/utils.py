# utils.py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F


def sanitize_cfg(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for sub_k, sub_v in sanitize_cfg(v).items():
                out[f"{k}.{sub_k}"] = sub_v
        elif isinstance(v, list):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def gray_2_colormap_np(img, cmap="rainbow", max=None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img < 0] = 0
    mask_invalid = img < 1e-10
    if max is None:
        img = img / (img.max() + 1e-8)
    else:
        img = img / (max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap


def rgb_tensor_to_np(img_tensor):
    img = img_tensor.cpu().detach().float()
    if img.shape[0] == 1:
        # Grayscale — repeat to 3 channels so loggers render consistently
        img = img.repeat(3, 1, 1)
    img = img.permute(1, 2, 0).numpy()          # (H, W, 3)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

#####################################
### RELATED WITH SELF SUPERVISED LOSS
#####################################

def warp_image(img, disp, padding="zeros"):
    """
    Idea: Warp right image into the left view using left-referenced disparity.
    This function works for both right-to-left and left-to-right (flipped for model needs).
    Not need to differentiate between left and right.
    """

    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    assert img.dim() == 4 and disp.dim() == 4 # tensors both of them 
    assert disp.shape == (B, 1, H, W)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij"
    )

    xx = xx[None, None, ...].expand(B, -1, -1, -1)
    yy = yy[None, None, ...].expand(B, -1, -1, -1)

    x_right = xx - disp 
    y_right = yy 

    valid = (
        (x_right >= 0) & (x_right <= (W - 1)) & 
        (y_right >= 0) & (y_right <= (H - 1)) & 
        torch.isfinite(x_right)
    )
    valid = valid.to(img.dtype)

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
            img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    return warped, valid


def lr_consistency_mask(disp_lr, disp_rl, threshold=1.0):
    disp_rl_warped, _ = warp_image(disp_rl, disp_lr)
    diff  = (disp_lr - disp_rl_warped).abs()
    mask  = (diff < threshold).float()
    return mask, diff


def compute_rl_disparity(model, left, right, iters):
    """
    Compute right-to-left disparity by running MonSter++ on horizontally
    flipped images, then flipping the output back to original coordinates.

    The model always receives a geometrically valid (left, right) stereo pair
    — flipping right→left swap ensures this — and the output is flipped back
    so disp_rl is in the same coordinate system as disp_lr.
    """
    assert left.ndim == 4 and right.ndim == 4, "Images must have 4 dimensions (B, C, H, W)"
    assert left.shape == right.shape, f"Shape discrepancies: {left.shape} vs {right.shape}"

    right_flipped = torch.flip(right, dims=[-1])
    left_flipped  = torch.flip(left,  dims=[-1])

    output = model(right_flipped, left_flipped, iters=iters)
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("Expected model output to contain init and iterative disparities")
    disp_init_flip, disp_preds_flip = output[0], output[1]

    disp_init_rl  = torch.flip(disp_init_flip, dims=[-1])
    disp_preds_rl = [torch.flip(d, dims=[-1]) for d in disp_preds_flip]

    return disp_init_rl, disp_preds_rl


def zbuffer_occlusion_mask(disp_lr):
    """
    Compute occlusion mask using z-buffer rendering — no RL pass needed.

    Projects left disparity into right view, keeps the frontmost pixel
    (largest disparity wins) at each destination, then marks which
    left pixels survived the z-buffer test.

    Args:
        disp_lr: (B, 1, H, W) disparity in left-view coordinates

    Returns:
        mask: (B, 1, H, W) float — 1=visible, 0=occluded
    """
    B, _, H, W = disp_lr.shape
    device = disp_lr.device
    dtype  = disp_lr.dtype

    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x[None, None].expand(B, 1, H, W)

    # Destination x in right view (rounded to nearest pixel)
    x_dst = torch.round(x - disp_lr).long()

    # Row indices (y doesn't change in rectified stereo)
    row_idx = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

    mask = torch.zeros(B, 1, H, W, device=device, dtype=dtype)

    for b in range(B):
        xd = x_dst[b, 0]           # (H, W) destination x
        d  = disp_lr[b, 0]         # (H, W) disparity values
        ri = row_idx[b, 0]         # (H, W) row indices

        # In-bounds pixels only
        inb = (xd >= 0) & (xd < W)

        if not inb.any():
            continue

        # Flat index in right view: row * W + col
        tgt_flat  = (ri[inb] * W + xd[inb]).view(-1)
        disp_flat = d[inb].view(-1)

        # Z-buffer: keep max disparity at each destination
        zbuf = torch.full((H * W,), -float('inf'), device=device, dtype=dtype)
        zbuf.scatter_reduce_(0, tgt_flat, disp_flat, reduce='amax')

        # Check: did my disparity win at my destination?
        # Look up the z-buffer value at each source pixel's destination
        zbuf_at_dst = torch.full_like(d, -float('inf'))
        zbuf_at_dst[inb] = zbuf[tgt_flat].view_as(d[inb])

        # Visible = my disparity matches the winner (within 0.5px tolerance)
        mask[b, 0] = ((d - zbuf_at_dst).abs() < 0.5).float() * inb.float()

    return mask
