from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Union

import torch

from ._paths import ensure_monster_paths


# ---------- small helpers ----------

def _device_from_maybe_str(d: Optional[Union[str, torch.device]]) -> torch.device:
    if d is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


# ---------- defaults ----------

DEFAULT_MONSTER_ARGS = SimpleNamespace(
    encoder="vitl",
    hidden_dims=[128, 128, 128],
    corr_implementation="reg",
    shared_backbone=False,
    corr_levels=2,
    corr_radius=4,
    n_downsample=2,
    n_gru_layers=3,
    max_disp=192,
    mixed_precision=False,
    slow_fast_gru=False,
)

# ---------- builders ----------


def build_monster(
    monster_ckpt: str,
    depth_anything_v2_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    args: SimpleNamespace = DEFAULT_MONSTER_ARGS,
    eval_only: bool = True,
) -> torch.nn.Module:
    """
    Build MonSter with third-party paths managed in one place.
    """
    device = _device_from_maybe_str(device)
    if depth_anything_v2_path is not None:
        try:
            args.depth_anything_v2_path = depth_anything_v2_path
        except Exception:
            if getattr(args, "depth_anything_v2_path", None) != depth_anything_v2_path:
                raise
    ensure_monster_paths()

    from core.monster import Monster  # MonSter uses top-level "core.*" imports

    model = Monster(args)

    if device.type == "cuda" and torch.cuda.device_count() > 1 and device.index is None:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)

    model.to(device)
    if monster_ckpt:
        state = torch.load(monster_ckpt, map_location=device)
        state = state["state_dict"] if "state_dict" in state else state
        if isinstance(model, torch.nn.DataParallel):
            if not any(k.startswith("module.") for k in state.keys()):
                state = {f"module.{k}": v for k, v in state.items()}
        else:
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    if eval_only:
        model.eval()
    return model



def __getattr__(name: str):
    if name == "MonsterInputPadder":
        ensure_monster_paths()
        from core.utils.utils import InputPadder as MonsterInputPadder

        globals()["MonsterInputPadder"] = MonsterInputPadder
        return MonsterInputPadder
    if name == "FsInputPadder":
        from .FoundationStereo.fs_core.utils.utils import InputPadder as FsInputPadder

        globals()["FsInputPadder"] = FsInputPadder
        return FsInputPadder
    raise AttributeError(f"module {__name__} has no attribute {name}")
