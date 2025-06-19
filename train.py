#!/usr/bin/env python3
"""
akioi-2048 DQN trainer  ·  CPU / Apple-Silicon
  · 保持原有训练逻辑、参数、CheckpointCallback
  · 上传改用 huggingface_hub HTTP API（无视频）
"""

from __future__ import annotations
import argparse, platform, multiprocessing as mp, time, warnings
from pathlib import Path

import torch, gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from huggingface_sb3 import load_from_hub  # 仍可用于 --resume
from huggingface_hub import create_repo, upload_file, HfApi

from akioi_gym import Akioi2048Env
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# ───────── 设备选择：MPS or CPU ─────────
DEVICE = (
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"▶ Using device: {DEVICE}")

# ───────── Windows/macOS 多进程守护 ──────
if platform.system() in {"Windows", "Darwin"}:
    mp.set_start_method("spawn", force=True)


# ───────── Gym VecEnv helper ────────────
def build_env(n_envs: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    return make_vec_env(Akioi2048Env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


# ───────── 创建或加载模型 ────────────────
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


# ────────── main ────────────────────────
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--n-envs", type=int, default=8)
    argp.add_argument("--timesteps", type=int, default=2_000_000)
    argp.add_argument("--checkpoint-freq", type=int, default=250_000)
    argp.add_argument("--resume", help="local .zip or HF repo")
    argp.add_argument("--hf-repo", help="huggingface repo id (optional)")
    args = argp.parse_args()

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
    local_zip = f"dqn_akioi2048_{stamp}.zip"
    model.save(local_zip)
    print(f"✓ Saved → {local_zip}")

    # ───────── 上传到 Hugging Face（可选） ─────────
    if args.hf_repo:
        print(f"↑ Uploading {local_zip} to https://huggingface.co/{args.hf_repo}")
        token = HfApi().token  # 需要提前 `huggingface-cli login`
        create_repo(repo_id=args.hf_repo, exist_ok=True, token=token)

        upload_file(
            repo_id=args.hf_repo,
            path_or_fileobj=local_zip,
            path_in_repo=local_zip,  # 保持同名
            token=token,
            commit_message=f"+{args.timesteps:,} steps on {time.ctime()}",
        )
        print("✔ Upload done")

    env.close()


if __name__ == "__main__":
    main()
