"""
Evaluate trained agent (CUDA / MPS / CPU)
"""

from __future__ import annotations
import argparse, os, platform, multiprocessing as mp
import numpy as np, torch
from stable_baselines3 import DQN
from huggingface_sb3 import load_from_hub
from akioi_gym import Akioi2048Env


# ─── device 选择逻辑：仅 MPS 或 CPU ──────────────────────────────
def detect_device() -> str:
    """
    Return "mps" on Apple Silicon, otherwise "cpu".
    CUDA is intentionally ignored even if available.
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = device()
print(f"▶ Eval device: {DEVICE}")


def load_model(src: str, env):
    if os.path.isfile(src):
        return DQN.load(src, env=env, device=DEVICE)
    return load_from_hub(src, "model.zip", env=env, device=DEVICE)


def main():
    if platform.system() in {"Windows", "Darwin"}:
        mp.set_start_method("spawn", force=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path.zip or HF repo[@commit]")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    env = Akioi2048Env()
    model = load_model(args.model, env)

    scores = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        tot = 0
        while not done:
            act, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _, _ = env.step(act)
            tot += reward
        scores.append(tot)
        print(f"Episode {ep + 1:3d}: {tot}")

    print("-" * 42)
    print(f"Mean score over {args.episodes} eps: {np.mean(scores):.1f}")


if __name__ == "__main__":
    main()
