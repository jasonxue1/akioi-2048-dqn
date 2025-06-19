"""
Universal trainer ── 支持 CUDA / Apple M-series / 纯 CPU (Win & macOS-Intel)

用法示例：
  # Intel / Windows（CPU）
  python train.py --n-envs 8 --timesteps 1_000_000

  # Apple M4（MPS）
  python train.py --n-envs 4 --timesteps 2_000_000 --hf-repo jasonxue1/akioi-2048-dqn
"""

from __future__ import annotations
import argparse, os, sys, time, platform, multiprocessing as mp
from pathlib import Path
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from huggingface_sb3 import load_from_hub, package_to_hub
from akioi_gym import Akioi2048Env

import warnings, urllib3
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


# ───────────────────────── 设备检测 ─────────────────────────── #
# ─── device 选择逻辑：仅 MPS 或 CPU ──────────────────────────────
def detect_device() -> str:
    """
    Return "mps" on Apple Silicon, otherwise "cpu".
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = detect_device()
print(f"▶ Using device: {DEVICE}")


# ──────────────────── Multiprocessing guard ─────────────────── #
def init_mp():
    if platform.system() in {"Windows", "Darwin"}:
        mp.set_start_method("spawn", force=True)


# ──────────────────── Env & Model helpers ───────────────────── #
def build_env(n_envs: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    return make_vec_env(Akioi2048Env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


def load_or_create(resume: str | None, env):
    if resume:
        if Path(resume).is_file():
            print(f"▶ Loading local checkpoint: {resume}")
            return DQN.load(resume, env=env, device=DEVICE)
        print(f"▶ Loading from Hugging Face: {resume}")
        return load_from_hub(resume, "model.zip", env=env, device=DEVICE)

    print("▶ Creating new model")
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=1_000_000,
        batch_size=1024,
        target_update_interval=10_000,
        verbose=1,
        tensorboard_log="./tb",
        device=DEVICE,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )


# ─────────────────────────── main ───────────────────────────── #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--checkpoint-freq", type=int, default=250_000)
    p.add_argument("--resume", type=str, default=None, help="path.zip or HF repo")
    p.add_argument("--hf-repo", type=str, default=None, help="push repo id")
    args = p.parse_args()

    init_mp()
    env = build_env(args.n_envs)
    model = load_or_create(args.resume, env)

    ckpt = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path="./checkpoints",
        name_prefix="dqn2048",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=CallbackList([ckpt]),
    )

    stamp = int(time.time())
    local_name = f"dqn_akioi2048_{stamp}"
    model.save(local_name)
    print(f"✓ Saved to {local_name}.zip")

    if args.hf_repo:
        print(f"↑ Uploading to {args.hf_repo}")
        package_to_hub(
            model=model,
            model_name=local_name,
            repo_id=args.hf_repo,
            commit_message=f"+{args.timesteps:,} steps on {time.ctime()}",
            env=env,
            model_architecture="DQN",
            task="Reinforcement-Learning",
            device=DEVICE,
            video_fps=5,
        )


if __name__ == "__main__":
    main()
