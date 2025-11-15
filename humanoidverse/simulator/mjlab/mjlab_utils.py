import os
from typing import List

import torch


def numpy_to_torch(x, device: str):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.tensor(x, device=device)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    return q[..., [1, 2, 3, 0]]


def reorder_second_axis(x: torch.Tensor, index_map: torch.Tensor) -> torch.Tensor:
    """Reorder a [B, N, ...] tensor along axis=1 using index_map (LongTensor)."""
    return x.index_select(1, index_map)


def normalize_name_for_mjlab(name: str, model, obj_type) -> str:
    """
    Best-effort normalization helper. For MuJoCo scene graphs that prefix names (e.g., "robot/"),
    try alternative variants so that mj_name2id can resolve them.
    """
    try_names: List[str] = [name]
    if not name.startswith("robot/"):
        try_names.append(f"robot/{name}")
    import mujoco
    for candidate in try_names:
        jid = mujoco.mj_name2id(model, obj_type, candidate)
        if jid >= 0:
            return candidate
    # fallback to original
    return name

