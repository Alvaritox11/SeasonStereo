from .sequence_loss import sequence_loss
from .self_supervised import self_supervised_sequence_loss


def build_loss(loss_cfg):
    loss_name = loss_cfg.name
    registry = {
        "sequence_loss":                sequence_loss,
        "self_supervised_sequence_loss": self_supervised_sequence_loss,
    }
    if loss_name not in registry:
        raise ValueError(
            f"Unknown loss '{loss_name}'. Available: {list(registry.keys())}"
        )

    loss_fn = registry[loss_name]
    params  = dict(loss_cfg.get("params", {}))

    if loss_name == "sequence_loss":
        def wrapped_loss(disp_preds, disp_init_pred, disp_gt, valid):
            return loss_fn(
                disp_preds     = disp_preds,
                disp_init_pred = disp_init_pred,
                disp_gt        = disp_gt,
                valid          = valid,
                **params,
            )
        wrapped_loss.is_self_supervised = False
        wrapped_loss.needs_rl           = False
        return wrapped_loss

    # ── Self-supervised loss ──────────────────────────────────────────────────
    if loss_name == "self_supervised_sequence_loss":
        # RL pass only needed when photometric is active AND using lr_consistency
        _needs_rl = (
            params.get("lambda_photo", 1.0) > 0 and
            params.get("occlusion_mode", "zbuffer") == "lr_consistency"
        )
        def wrapped_loss(
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
            left_building_mask=None,
            right_building_mask=None,
        ):
            return loss_fn(
                left              = left,
                right             = right,
                left_real         = left_real,
                right_real        = right_real,
                disp_init_pred_lr = disp_init_pred_lr,
                disp_preds_lr     = disp_preds_lr,
                disp_init_pred_rl = disp_init_pred_rl,
                disp_preds_rl     = disp_preds_rl,
                disp_gt           = disp_gt,
                valid             = valid,
                left_building_mask = left_building_mask,
                right_building_mask = right_building_mask,
                **params,
            )

        wrapped_loss.is_self_supervised = True
        wrapped_loss.needs_rl           = _needs_rl

        # Print active terms for clarity at launch
        active = []
        if params.get("lambda_photo",  1.0) > 0: active.append(f"photo={params.get('lambda_photo',  1.0)}")
        if params.get("lambda_pseudo", 0.5) > 0: active.append(f"pseudo={params.get('lambda_pseudo', 0.5)}")
        if params.get("lambda_smooth", 0.1) > 0: active.append(f"smooth={params.get('lambda_smooth', 0.1)}")
        occ = params.get("occlusion_mode", "zbuffer")
        photo_buildings = params.get("photometric_on_buildings_only", False)
        print(f"SS loss active terms: {', '.join(active) or 'NONE'}")
        print(f"Occlusion mode: {occ}")
        print(f"Photometric on buildings only: {'YES' if photo_buildings else 'NO'}")
        print(f"RL forward pass:   {'YES' if _needs_rl         else 'SKIPPED'}")
        return wrapped_loss
