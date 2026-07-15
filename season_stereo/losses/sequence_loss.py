import torch
import torch.nn.functional as F

def sequence_loss(
    disp_preds,
    disp_init_pred,
    disp_gt,
    valid,
    loss_gamma=0.9,
    max_disp=192,
    ignore_border=32,
):
    """Loss function defined over sequence of flow predictions"""
    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    # valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    valid = (valid >= 0.5).contiguous() & (mag < max_disp).contiguous().unsqueeze(1)
    # Ignore borders of the image where cropped things might get invalid predictions
    if ignore_border > 0:
        # top / bottom (height)
        valid[:, :, :ignore_border, :] = False
        valid[:, :, -ignore_border:, :] = False
        # left / right (width)
        valid[..., :ignore_border] = False
        valid[..., -ignore_border:] = False
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid.bool()]).any()

    init_valid = valid.bool() & ~torch.isnan(disp_init_pred)
    disp_loss += 1.0 * F.smooth_l1_loss(
        disp_init_pred[init_valid], disp_gt[init_valid], reduction="mean"
    )

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape,
            valid.shape,
            disp_gt.shape,
            disp_preds[i].shape,
        ]
        disp_loss += i_weight * i_loss[valid.bool() & ~torch.isnan(i_loss)].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        "train/epe": epe.mean(),
        "train/1px": (epe < 1).float().mean(),
        "train/3px": (epe < 3).float().mean(),
        "train/5px": (epe < 5).float().mean(),
    }
    return disp_loss, metrics