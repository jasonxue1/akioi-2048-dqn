from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ACTION_MEANINGS: tuple[str, ...] = ("Down", "Right", "Up", "Left")  # 0..3


def _build_policy(state_dict: dict, data_cfg: dict) -> nn.Module:
    """Rebuild the MLP policy network that matches saved weights."""
    # hidden sizes from data file
    arch_raw = data_cfg["policy_kwargs"]["net_arch"]
    hidden = [
        int(x) for x in (arch_raw[0] if isinstance(arch_raw[0], list) else arch_raw)
    ]

    in_dim = 16  # 4×4 board flattened
    n_actions = 4  # fixed for 2048

    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, n_actions))

    policy = nn.Sequential(*layers)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()
    return policy


def choose_move(
    model_dir: str | Path,
    board: list[list[int]] | np.ndarray,
    *,
    deterministic: bool = True,
) -> tuple[int, str]:
    """
    Parameters
    ----------
    model_dir : folder containing `policy.pth` and `data`
    board     : 4×4 current board (list[list[int]] or np.ndarray)
    deterministic : True → argmax, False → sample from softmax

    Returns
    -------
    action_id : 0 – 3
    direction : 'Down'|'Right'|'Up'|'Left'
    """
    model_dir = Path(model_dir)
    data_cfg = json.loads((model_dir / "data").read_text())
    state_dict = torch.load(model_dir / "policy.pth", map_location="cpu")

    policy = _build_policy(state_dict, data_cfg)

    obs = np.asarray(board, dtype=np.float32).reshape(1, -1)  # shape (1,16)
    with torch.no_grad():
        logits = policy(torch.from_numpy(obs))
    if deterministic:
        action_id = int(torch.argmax(logits, dim=1))
    else:
        probs = torch.softmax(logits, dim=1)
        action_id = int(torch.multinomial(probs, 1))

    return action_id, ACTION_MEANINGS[action_id]


# ---------------- quick demo ----------------
if __name__ == "__main__":
    SAMPLE_BOARD = [
        [2, 4, 2, 4],
        [2, 4, 2, 4],
        [2, 4, 2, 4],
        [0, 0, 0, 0],
    ]
    act, direction = choose_move(
        model_dir="model", board=SAMPLE_BOARD, deterministic=False
    )
    print(f"Recommended move: {direction} (id={act})")
