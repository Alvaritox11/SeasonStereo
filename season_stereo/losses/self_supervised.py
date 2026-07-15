import torch
import torch.nn.functional as F

from utils.utils import *


def build_valid_mask(disp_gt, valid, max_disp, ignore_border):
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5).contiguous() & (mag < max_disp).contiguous().unsqueeze(1)

    if ignore_border > 0:
        # top / bottom (height)
        valid[:, :, :ignore_border, :] = False
        valid[:, :, -ignore_border:, :] = False
        # left / right (width)
        valid[..., :ignore_border]      = False
        valid[..., -ignore_border:]     = False
        
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    return valid  # (B, 1, H, W) bool


def ssim_loss_map(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Per-pixel SSIM loss map (1 - SSIM) / 2, averaged over channels.

    Returns: (B, 1, H, W)
    """
    pad = window_size // 2

    mu_x  = F.avg_pool2d(x, window_size, 1, pad)
    mu_y  = F.avg_pool2d(y, window_size, 1, pad)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y

    sig_x  = F.avg_pool2d(x * x, window_size, 1, pad) - mu_x2
    sig_y  = F.avg_pool2d(y * y, window_size, 1, pad) - mu_y2
    sig_xy = F.avg_pool2d(x * y, window_size, 1, pad) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    ssim_map = num / den.clamp(min=1e-8)

    return ((1 - ssim_map) / 2).mean(dim=1, keepdim=True)  # (B, 1, H, W)


def photometric_map(left_real, right_real, valid_mask, disp_lr, disp_rl=None,
                    ssim_alpha=0.85, lr_threshold=1.0, occlusion_mode="zbuffer",
                    left_photo_mask=None, right_photo_mask=None):
    """
    Photometric loss with configurable occlusion masking. The model predicts disparity from any 
    combination of real/synthetic images, but reconstruction quality is always evaluated on the 
    real pair — which is appearance-consistent and free of seasonal differences.

    occlusion_mode:
        "zbuffer"        — z-buffer mask from disp_lr only, left reconstruction only, no RL needed
        "lr_consistency" — LR consistency mask, symmetric left+right reconstruction, needs disp_rl
    """
    left_n  = left_real  / 255.0
    right_n = right_real / 255.0

    # Left reconstruction: warp right -> left (always computed)
    left_rec, valid_warp_l = warp_image(right_n, disp_lr)
    ssim_l = ssim_loss_map(left_n, left_rec)
    l1_l   = (left_n - left_rec).abs().mean(dim=1, keepdim=True)
    photo_l_map = ssim_alpha * ssim_l + (1 - ssim_alpha) * l1_l

    if occlusion_mode == "zbuffer":
        occ_mask = zbuffer_occlusion_mask(disp_lr)
        mask_l = (valid_mask & valid_warp_l.bool() & occ_mask.bool()).float()
        if left_photo_mask is not None:
            mask_l = mask_l * left_photo_mask.float()
        photo_l = (photo_l_map * mask_l).sum() / mask_l.sum().clamp(min=1)
        return photo_l

    elif occlusion_mode == "lr_consistency":
        assert disp_rl is not None, "disp_rl required for lr_consistency mode"

        lr_mask, _ = lr_consistency_mask(disp_lr, disp_rl.detach(), threshold=lr_threshold)

        # Left direction
        mask_l = (valid_mask & lr_mask.bool() & valid_warp_l.bool()).float()
        if left_photo_mask is not None:
            mask_l = mask_l * left_photo_mask.float()
        photo_l = (photo_l_map * mask_l).sum() / mask_l.sum().clamp(min=1)

        # Right reconstruction: warp left -> right
        right_rec, valid_warp_r = warp_image(left_n, -disp_rl)
        ssim_r = ssim_loss_map(right_n, right_rec)
        l1_r   = (right_n - right_rec).abs().mean(dim=1, keepdim=True)
        photo_r_map = ssim_alpha * ssim_r + (1 - ssim_alpha) * l1_r

        mask_r = (valid_mask & lr_mask.bool() & valid_warp_r.bool()).float()
        if right_photo_mask is not None:
            mask_r = mask_r * right_photo_mask.float()
        photo_r = (photo_r_map * mask_r).sum() / mask_r.sum().clamp(min=1)

        return 0.5 * photo_l + 0.5 * photo_r

    else:
        raise ValueError(f"Unknown occlusion_mode: {occlusion_mode}")


def pseudo_gt_loss(pred, pseudo, valid_mask, reduction="mean", beta=1.0):
    mask = valid_mask.bool() & ~torch.isnan(pred - pseudo)
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return F.smooth_l1_loss(pred[mask], pseudo[mask], reduction=reduction, beta=beta)


def edge_aware_smoothness_map(disp, image):
    image_n = image / 255.0

    mean_disp = disp.mean(dim=[2, 3], keepdim=True).clamp(min=1e-7)
    disp_norm = disp / mean_disp

    grad_disp_x = (disp_norm[:, :, :, :-1] - disp_norm[:, :, :, 1:]).abs()
    grad_disp_y = (disp_norm[:, :, :-1, :] - disp_norm[:, :, 1:, :]).abs()

    grad_img_x = (image_n[:, :, :, :-1] - image_n[:, :, :, 1:]).abs().mean(dim=1, keepdim=True)
    grad_img_y = (image_n[:, :, :-1, :] - image_n[:, :, 1:, :]).abs().mean(dim=1, keepdim=True)

    return (grad_disp_x * torch.exp(-grad_img_x)).mean() + \
           (grad_disp_y * torch.exp(-grad_img_y)).mean()


def _prepare_photo_mask(mask, reference, name):
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"{name} shape {tuple(mask.shape)} does not match reference shape {tuple(reference.shape)}"
        )
    return mask.to(device=reference.device).bool()


def self_supervised_sequence_loss(
    left,
    right,
    left_real,
    right_real,
    disp_init_pred_lr,
    disp_preds_lr,
    disp_init_pred_rl,
    disp_preds_rl,
    disp_gt,
    valid,
    loss_gamma=0.9,
    max_disp=576,
    ignore_border=32,
    lambda_photo=1.0,
    lambda_pseudo=0.05,
    lambda_smooth=0.1,
    occlusion_mode="zbuffer",
    photometric_on_buildings_only=False,
    left_building_mask=None,
    right_building_mask=None,
):

    n_predictions = len(disp_preds_lr)
    assert n_predictions >= 1
    
    use_photo  = lambda_photo  > 0
    use_pseudo = lambda_pseudo > 0
    use_smooth = lambda_smooth > 0

    if use_photo:
        if occlusion_mode == "lr_consistency":
            assert disp_preds_rl is not None, "disp_preds_rl required for lr_consistency"
            assert len(disp_preds_rl) == n_predictions

    valid_mask = build_valid_mask(disp_gt, valid, max_disp, ignore_border)

    left_photo_mask = None
    right_photo_mask = None
    if use_photo and photometric_on_buildings_only:
        if left_building_mask is None:
            raise ValueError("photometric_on_buildings_only=True but left_building_mask was not provided")
        if occlusion_mode == "lr_consistency" and right_building_mask is None:
            raise ValueError("photometric_on_buildings_only=True with lr_consistency but right_building_mask was not provided")

        left_photo_mask = _prepare_photo_mask(left_building_mask, valid_mask, "left_building_mask")
        right_photo_mask = _prepare_photo_mask(right_building_mask, valid_mask, "right_building_mask")

    disp_loss = 0.0

    device = disp_gt.device
    loss_photo  = torch.tensor(0.0, device=device)
    loss_pseudo = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)

    # ── Init prediction — weight 1.0 ─────────────────────────────────────────
    init_valid = valid_mask.bool() & ~torch.isnan(disp_init_pred_lr)

    if use_photo:
        _photo = lambda_photo * photometric_map(
            left_real, right_real, init_valid, disp_init_pred_lr,
            disp_rl=disp_init_pred_rl,
            occlusion_mode=occlusion_mode,
            left_photo_mask=left_photo_mask,
            right_photo_mask=right_photo_mask,
        )
        disp_loss  += _photo
        loss_photo += _photo.detach()

    if use_pseudo:
        _pseudo = lambda_pseudo * pseudo_gt_loss(
            disp_init_pred_lr, disp_gt, init_valid
        )
        disp_loss   += _pseudo
        loss_pseudo += _pseudo.detach()

    if use_smooth:
        _smooth = lambda_smooth * edge_aware_smoothness_map(
            disp_init_pred_lr, left
        )
        disp_loss   += _smooth
        loss_smooth += _smooth.detach()

    # ── Per-iteration losses ──────────────────────────────────────────────────
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)

        disp_lr   = disp_preds_lr[i]
        disp_rl_i = disp_preds_rl[i] if disp_preds_rl is not None else None

        iter_loss = 0.0

        if use_photo:
            _photo = lambda_photo * photometric_map(
                left_real, right_real, valid_mask, disp_lr,
                disp_rl=disp_rl_i,
                occlusion_mode=occlusion_mode,
                left_photo_mask=left_photo_mask,
                right_photo_mask=right_photo_mask,
            )
            iter_loss  += _photo
            loss_photo += (i_weight * _photo).detach()

        if use_pseudo:
            _pseudo = lambda_pseudo * pseudo_gt_loss(
                disp_lr, disp_gt, valid_mask
            )
            iter_loss   += _pseudo
            loss_pseudo += (i_weight * _pseudo).detach()

        if use_smooth:
            _smooth = lambda_smooth * edge_aware_smoothness_map(
                disp_lr, left
            )
            iter_loss   += _smooth
            loss_smooth += (i_weight * _smooth).detach()

        disp_loss += i_weight * iter_loss

    final_pred = disp_preds_lr[-1]
    epe        = (final_pred - disp_gt).abs().squeeze(1)
    epe_valid  = epe.view(-1)[valid_mask.view(-1)]

    if not valid_mask.any():
        epe_valid = torch.tensor([0.0], device=final_pred.device)

    metrics = {
        "train/epe":        epe_valid.mean(),
        "train/1px":        (epe_valid < 1).float().mean(),
        "train/3px":        (epe_valid < 3).float().mean(),
        "train/5px":        (epe_valid < 5).float().mean(),
        "train/valid_pct":  valid_mask.float().mean(),
        "train/loss_photo": loss_photo,
        "train/loss_pseudo":loss_pseudo,
        "train/loss_smooth":loss_smooth,
    }

    if left_photo_mask is not None:
        metrics["train/photo_building_pct"] = left_photo_mask.float().mean()

    return disp_loss, metrics
