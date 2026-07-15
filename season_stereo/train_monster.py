"""Train the MonSter stereo model on rectified synthetic/real pairs."""

import os
from datetime import datetime
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

import thirdparty

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from accelerate.logging import get_logger
from accelerate.tracking import TensorBoardTracker, WandBTracker

from utils.utils import *
from helpers import *
from losses import build_loss
from datasets.augmentations import StereoAugmentor
from datasets.dataset_stereo import StereoDFC


QUAL_PAIR_SYNC  = "JAX_031_005_RGB-JAX_031_004_RGB"
QUAL_PAIR_DIACH = "OMA_281_006_RGB-OMA_281_016_RGB"


def unpack_model_output(output):
    """Return the stereo outputs even if a legacy model also returns extras."""
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        raise ValueError("Expected model output to contain at least init and iterative disparities")
    return output[0], output[1]


def fetch_optimizer(args, model):
    """Create optimizer and scheduler."""
    unwrapped = getattr(model, "module", model)
    feat_decoder = getattr(unwrapped, "feat_decoder", None)

    if feat_decoder is not None:
        feat_decoder_params = list(feat_decoder.parameters())
        feat_decoder_ids = {id(p) for p in feat_decoder_params}
        rest_params = [
            p for p in model.parameters()
            if id(p) not in feat_decoder_ids and p.requires_grad
        ]
        params_dict = [
            {"params": feat_decoder_params, "lr": args.lr / 2.0},
            {"params": rest_params, "lr": args.lr},
        ]
        scheduler_lrs = [args.lr / 2.0, args.lr]
    else:
        params_dict = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        scheduler_lrs = args.lr

    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        scheduler_lrs,
        args.max_step + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )
    return optimizer, scheduler


def load_dataset(cfg, mode="train", transforms=None):
    """
    Build a StereoDFC dataset for train or val.
 
    Train data:  {root}/train/  +  cfg.dfc.experiment_csv
    Val data:    {root}/val/    +  {root}/experiments/sub_val.csv   (FIXED for all experiments)
    """
    if cfg.dataset != "dfc":
        raise ValueError(f"Unknown dataset {cfg.dataset}")
 
    is_train   = mode == "train"
    use_water  = cfg.dfc.use_water_masks
    use_tree   = cfg.dfc.use_tree_masks
    loss_params = cfg.loss.get("params", {})
    photo_on_buildings = (
        cfg.loss.name == "self_supervised_sequence_loss" and
        float(loss_params.get("lambda_photo", 1.0)) > 0 and
        bool(loss_params.get("photometric_on_buildings_only", False))
    )
    use_building = bool(cfg.dfc.get("use_building_masks", False)) or photo_on_buildings
    return_building = bool(cfg.dfc.get("return_building_masks", False)) or photo_on_buildings
 
    if is_train:
        data_root      = os.path.join(cfg.dfc.root, "train")
        disparity_dir  = cfg.dfc.get("disparity_dir", os.path.join(data_root, "disparity"))
        experiment_csv = cfg.dfc.experiment_csv
        aois_csv       = cfg.dfc.get("train_aois_csv", None)
    else:
        # Fixed validation set, the same for all the experiments
        data_root      = os.path.join(cfg.dfc.root, "val")
        disparity_dir  = cfg.dfc.get("val_disparity_dir", os.path.join(data_root, "disparity"))
        experiment_csv = os.path.join(cfg.dfc.root, "experiments", "sub_val.csv")
        aois_csv       = cfg.dfc.get("val_aois_csv", None)
 
    def _mask_dir(subdir):
        return os.path.join(data_root, subdir)
 
    return StereoDFC(
        left_dir              = _mask_dir("L"),
        right_dir             = _mask_dir("R"),
        disparity_dir         = disparity_dir,
        experiment_csv        = experiment_csv,
        train                 = is_train,
        crop_size             = cfg.image_size,
        transforms            = transforms,
        left_masks_dir        = _mask_dir("masks/L"),
        right_masks_dir       = _mask_dir("masks/R"),
        left_water_masks_dir  = _mask_dir("water_masks/L") if use_water else None,
        right_water_masks_dir = _mask_dir("water_masks/R") if use_water else None,
        left_tree_masks_dir   = _mask_dir("tree_masks/L")  if use_tree  else None,
        right_tree_masks_dir  = _mask_dir("tree_masks/R")  if use_tree  else None,
        left_building_masks_dir  = _mask_dir("building_masks/L") if use_building else None,
        right_building_masks_dir = _mask_dir("building_masks/R") if use_building else None,
        aois_csv              = aois_csv,
        image_ext             = cfg.dfc.image_ext,
        disparity_ext         = cfg.dfc.disparity_ext,
        mask_ext              = cfg.dfc.mask_ext,
        return_geometry_masks = cfg.dfc.return_geometry_masks,
        return_water_masks    = cfg.dfc.return_water_masks,
        mask_out_water        = cfg.dfc.mask_out_water,
        return_tree_masks     = cfg.dfc.return_tree_masks,
        mask_out_trees        = cfg.dfc.mask_out_trees,
        return_building_masks = return_building,
    )


def run_validation(model, val_loader, accelerator, tb_tracker, wandb_tracker, total_step, cfg):
    model.eval()

    counters = {
        split: {"n": 0, "epe": 0.0, "d1_1px": 0.0, "d1_3px": 0.0}
        for split in ("all", "synchronic", "diachronic")
    }
 
    qual_samples = {"synchronic": None, "diachronic": None}
    qual_targets = {
        "synchronic": QUAL_PAIR_SYNC,
        "diachronic": QUAL_PAIR_DIACH,
    }
 
    for val_data in tqdm(val_loader, dynamic_ncols=True,
                         disable=not accelerator.is_main_process):
        val_left     = val_data["left"]  * 255.0
        val_right    = val_data["right"] * 255.0
        val_disp_gt  = val_data["disparity"]
        val_valid    = val_data.get(
            "valid", (val_disp_gt > 0).float()
        ).to(val_disp_gt.device)
        val_filename = val_data["filename"][0]
 
        diach_flag = val_data.get("diachronic", None)
        if diach_flag is not None:
            val_split = "diachronic" if bool(diach_flag[0].item()) else "synchronic"
        else:
            val_split = None
 
        padder = thirdparty.MonsterInputPadder(val_left.shape, divis_by=32)
        left_p, right_p = padder.pad(val_left, val_right)
 
        with torch.no_grad():
            disp_pred = model(left_p, right_p,
                              iters=cfg.valid_iters, test_mode=True)
 
        disp_pred = padder.unpad(disp_pred)
 
        valid_mask = val_valid >= 0.5
        epe_map    = torch.abs(disp_pred - val_disp_gt)
        epe_valid  = epe_map[valid_mask]
 
        if epe_valid.numel() == 0:
            continue
 
        epe_mean, out_1px, out_3px = accelerator.gather_for_metrics((
            epe_valid.mean().unsqueeze(0),
            (epe_valid > 1.0).float().mean().unsqueeze(0),
            (epe_valid > 3.0).float().mean().unsqueeze(0),
        ))
 
        epe_val  = epe_mean.mean().item()
        d1_1_val = out_1px.mean().item()
        d1_3_val = out_3px.mean().item()
        n        = epe_mean.numel()
 
        for key in (["all", val_split] if val_split else ["all"]):
            counters[key]["n"]      += n
            counters[key]["epe"]    += epe_val  * n
            counters[key]["d1_1px"] += d1_1_val * n
            counters[key]["d1_3px"] += d1_3_val * n
 
        # Capture specific qualitative pairs
        for split_name, target in qual_targets.items():
            if val_filename == target and qual_samples[split_name] is None:
                qual_samples[split_name] = {
                    "disp_pred": disp_pred[0].detach().cpu().squeeze(),  # (H, W)
                    "disp_gt":   val_disp_gt[0].detach().cpu().squeeze(),
                }
 
    # ── Log metrics ───────────────────────────────────────────────────────────
    if accelerator.is_main_process:
 
        log_dict = {}
        for key, c in counters.items():
            n = c["n"]
            if n == 0:
                continue
            log_dict[f"val/epe_{key}"]    = c["epe"]    / n
            log_dict[f"val/d1_1px_{key}"] = 100 * c["d1_1px"] / n
            log_dict[f"val/d1_3px_{key}"] = 100 * c["d1_3px"] / n
 
        accelerator.log(log_dict, total_step)
 
        # ── Summary table ─────────────────────────────────────────────────
        print(f"\n{'─'*65}")
        print(f"  Validation @ step {total_step}")
        print(f"{'─'*65}")
        print(f"  {'split':12s}  {'EPE':>8}  {'D1>1px':>8}  {'D1>3px':>8}  {'n':>5}")
        for key in ("all", "synchronic", "diachronic"):
            c = counters[key]
            n = c["n"]
            if n == 0:
                print(f"  {key:12s}  {'—':>8}  {'—':>8}  {'—':>8}  {0:>5}")
                continue
            print(
                f"  {key:12s}"
                f"  {c['epe']/n:8.4f}"
                f"  {100*c['d1_1px']/n:7.2f}%"
                f"  {100*c['d1_3px']/n:7.2f}%"
                f"  {n:>5}"
            )
        print(f"{'─'*65}")
 
        # ── Qualitative disparity logging ─────────────────────────────────
        if total_step % cfg.logging.image_log_frequency == 0:
            images_dict = {}
            for split_name, q in qual_samples.items():
                if q is None:
                    continue
                pred = q["disp_pred"]
                gt   = q["disp_gt"]
                shared_max = max(pred.max().item(), gt.max().item())
                prefix = f"val_{split_name}"
                images_dict[f"{prefix}/disp_pred"] = gray_2_colormap_np(
                    pred, cmap="plasma", max=shared_max
                )
                images_dict[f"{prefix}/disp_gt"] = gray_2_colormap_np(
                    gt, cmap="plasma", max=shared_max
                )
 
            if images_dict:
                log_images(accelerator, tb_tracker, wandb_tracker,
                           images_dict, total_step)
    
    epe_all = counters["all"]["epe"] / counters["all"]["n"] if counters["all"]["n"] > 0 else float("inf")
    return epe_all

 


@hydra.main(
    version_base=None,
    config_path="training_configs",
    config_name="exp4-pseudoGT_0.05-photo_0.1_buildings-smooth_0.1",
)
def main(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg.logging.run_name or timestamp

    run_dir = Path(cfg.logdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    logger = get_logger(__name__)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    data_loader_config = DataLoaderConfiguration(use_seedable_sampler=True, data_seed=cfg.seed)

    trackers = []
    if "tensorboard" in cfg.logging.trackers:
        trackers.append(TensorBoardTracker(run_name=run_name, logging_dir=str(run_dir)))

    if "wandb" in cfg.logging.trackers:
        trackers.append(
            WandBTracker(
                run_name=cfg.logging.project_name,
                name=run_name,
                entity=cfg.logging.wandb.entity,
                dir=str(run_dir),
                mode=cfg.logging.wandb.mode,
            )
        )

    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=data_loader_config,
        log_with=trackers if len(trackers) > 0 else None,
        project_dir=str(run_dir),
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
    )

    accelerator.init_trackers(
        project_name=cfg.logging.project_name,
        config=sanitize_cfg(OmegaConf.to_container(cfg, resolve=True)),
    )

    tb_tracker = get_tb_tracker(accelerator)
    wandb_tracker = get_wandb_tracker(accelerator)

    with open(run_dir / "hparams.yml", "w") as f:
        yaml.safe_dump(sanitize_cfg(OmegaConf.to_container(cfg, resolve=True)), f)

    augmentor = StereoAugmentor(**cfg.augmentation)
    skip_validation = bool(cfg.get("skip_validation", False))

    train_dataset = load_dataset(cfg, mode="train", transforms=augmentor)

    if skip_validation:
        val_dataset = None
        print(f"Train: {len(train_dataset)} samples | Val: skipped")
    else:
        val_dataset = load_dataset(cfg, mode="val", transforms=None)
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples")

        val_stems = {s["stem"] for s in val_dataset.samples}
        for stem in (QUAL_PAIR_SYNC, QUAL_PAIR_DIACH):
            tag = "✓" if stem in val_stems else "✗ NOT FOUND"
            print(f"  Qual pair {tag}: {stem}")


    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // cfg.num_gpu,
        pin_memory=True,
        shuffle=True,
        num_workers=8,
        drop_last=True,
    )
    val_loader = None
    if not skip_validation:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
        )

    if cfg.restore_ckpt is not None:
        assert cfg.restore_ckpt.endswith(".pth")
        assert os.path.exists(cfg.restore_ckpt)
        print(f"Loading checkpoint from {cfg.restore_ckpt}")

    model = thirdparty.build_monster(
        monster_ckpt=cfg.restore_ckpt,
        depth_anything_v2_path=cfg.depth_anything_v2_path,
        device="cpu",
        args=cfg,
        eval_only=False,
    )

    if cfg.restore_ckpt is not None:
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")

    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    loss_fn = build_loss(cfg.loss)

    if skip_validation:
        train_loader, model, optimizer, lr_scheduler = accelerator.prepare(
            train_loader, model, optimizer, lr_scheduler
        )
    else:
        train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(
            train_loader, model, optimizer, lr_scheduler, val_loader
        )
    model.to(accelerator.device)

    best_epe = float("inf")
    total_step = 0
    while total_step < cfg.max_step:
        model.train()
        getattr(model, "module", model).freeze_bn()
 
        for data in tqdm(train_loader, dynamic_ncols=True,
                         disable=not accelerator.is_main_process):
            left  = data["left"]  * 255.0
            right = data["right"] * 255.0

            left_real  = data.get("left_real",  data["left"])  * 255.0
            right_real = data.get("right_real", data["right"]) * 255.0
 
            disp_gt    = data["disparity"]
            valid      = data.get("valid", (disp_gt > 0).float()).to(disp_gt.device)
            diachronic = data.get("diachronic", None)
            left_building_mask = data.get("left_building_mask", None)
            right_building_mask = data.get("right_building_mask", None)
 
            # ── Forward pass ──────────────────────────────────────────────
            if loss_fn.is_self_supervised:
                with accelerator.autocast():
                    disp_init_pred, disp_preds = unpack_model_output(
                        model(left, right, iters=cfg.train_iters)
                    )
                    if loss_fn.needs_rl:
                        disp_init_rl, disp_preds_rl = compute_rl_disparity(
                            model, left, right, iters=cfg.train_iters
                        )
                    else:
                        disp_init_rl  = None
                        disp_preds_rl = None

                loss, metrics = loss_fn(
                    left              = left,
                    right             = right,
                    left_real         = left_real,
                    right_real        = right_real,
                    left_building_mask = left_building_mask,
                    right_building_mask = right_building_mask,
                    disp_init_pred_lr = disp_init_pred,
                    disp_preds_lr     = disp_preds,
                    disp_init_pred_rl = disp_init_rl,
                    disp_preds_rl     = disp_preds_rl,
                    disp_gt           = disp_gt,
                    valid             = valid,
                )
            else:
                with accelerator.autocast():
                    disp_init_pred, disp_preds = unpack_model_output(
                        model(left, right, iters=cfg.train_iters)
                    )
                loss, metrics = loss_fn(
                    disp_preds     = disp_preds,
                    disp_init_pred = disp_init_pred,
                    disp_gt        = disp_gt,
                    valid          = valid,
                )
 
            if diachronic is not None:
                metrics["train/diachronic_ratio"] = diachronic.float().mean()
 
            # ── Backward & optimiser ──────────────────────────────────────
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
 
            total_step += 1
 
            accelerator.log(
                {
                    "train/loss":          accelerator.reduce(loss.detach(), "mean"),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                },
                step=total_step,
            )
            accelerator.log(accelerator.reduce(metrics, "mean"), step=total_step)
 
            # ── Train image logging ───────────────────────────────────────
            if total_step % 20 == 0 and accelerator.is_main_process:
                disp_pred_vis = disp_preds[-1][0].squeeze()
                disp_gt_vis   = disp_gt[0].squeeze()
                shared_max    = torch.max(disp_pred_vis.max(), disp_gt_vis.max()).item()
                log_images(accelerator, tb_tracker, wandb_tracker, {
                    "train/disp_pred":  gray_2_colormap_np(disp_pred_vis, cmap="plasma", max=shared_max),
                    "train/disp_gt":    gray_2_colormap_np(disp_gt_vis,   cmap="plasma", max=shared_max),
                    "train/right_img":  (right[0].detach().cpu().permute(1, 2, 0) / 255.0).clamp(0, 1).numpy(),
                }, total_step)
 
            # ── Validation ────────────────────────────────────────────────
            if not skip_validation and total_step % cfg.val_frequency == 0:
                epe_all = run_validation(
                    model, val_loader, accelerator,
                    tb_tracker, wandb_tracker,
                    total_step, cfg,
                )
                if accelerator.is_main_process and epe_all < best_epe:
                    best_epe = epe_all
                    torch.save(
                        accelerator.unwrap_model(model).state_dict(),
                        run_dir / "best.pth",
                    )
                    print(f"  New best EPE: {best_epe:.4f} — saved best.pth")
                model.train()
                getattr(model, "module", model).freeze_bn()
 
            if total_step >= cfg.max_step:
                break
 
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(),
                   run_dir / "final.pth")
 
    accelerator.end_training()



if __name__ == "__main__":
    main()
