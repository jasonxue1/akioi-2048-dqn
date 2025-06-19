from __future__ import annotations
import argparse, platform, multiprocessing as mp, time, warnings, os
from datetime import datetime
from pathlib import Path

import torch, gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from huggingface_sb3 import load_from_hub
from huggingface_hub import create_repo, upload_file, HfApi
from urllib3.exceptions import NotOpenSSLWarning

from akioi_gym import Akioi2048Env

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# ───────── 设备选择 ─────────
DEVICE = (
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"▶ Using device: {DEVICE}")

if platform.system() in {"Windows", "Darwin"}:
    mp.set_start_method("spawn", force=True)


# ───────── 构建环境 ─────────
def build_env(n_envs: int):
    from stable_baselines3.common.vec_env import SubprocVecEnv

    return make_vec_env(Akioi2048Env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


# ───────── 载入 / 创建模型 ─────────
def load_or_create(resume: str | None, env):
    if resume:
        if Path(resume).is_file():
            print(f"▶ Loading local checkpoint: {resume}")
            return DQN.load(resume, env=env, device=DEVICE)

        # ---------- HF repo ----------
        print(f"▶ Loading from Hugging Face: {resume}")
        # ① 下载文件，得到本地路径
        zip_path = load_from_hub(repo_id=resume, filename="model.zip")
        # ② 再交给 SB3 读取
        return DQN.load(zip_path, env=env, device=DEVICE)

    # ---------- 新模型 ----------
    print("▶ Creating new model")
    big_arch = [1024, 1024, 512]
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=1.5e-4,
        buffer_size=3_000_000,
        batch_size=1024,
        target_update_interval=10_000,
        verbose=1,
        tensorboard_log="./tb",
        device=DEVICE,
        policy_kwargs=dict(
            net_arch=big_arch,
            # dueling=True,
        ),
    )


# ──────────────── main ────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--checkpoint-freq", type=int, default=250_000)
    ap.add_argument("--resume", help=".zip 或 HF repo")
    ap.add_argument("--hf-repo", help="huggingface repo id (可选)")
    args = ap.parse_args()

    # --- run/<timestamp>/ 目录
    run_dir = Path("runs") / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(args.n_envs)
    model = load_or_create(args.resume, env)

    ckpt_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq // args.n_envs,
        save_path=str(ckpt_dir),
        name_prefix="dqn2048",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=False,
        callback=CallbackList([ckpt_cb]),
    )

    # --- 保存主权重到 run_dir
    local_zip = run_dir / f"dqn_akioi2048_{int(time.time())}.zip"
    model.save(str(local_zip))
    print(f"✓ Saved → {local_zip}")
    # --- 上传（可选）
    if args.hf_repo:
        token = HfApi().token  # 已 login
        create_repo(args.hf_repo, exist_ok=True, token=token)

        # 1) 历史归档版本
        upload_file(
            repo_id=args.hf_repo,
            path_or_fileobj=str(local_zip),
            path_in_repo=f"versions/{local_zip.name}",  # e.g. versions/dqn_...zip
            token=token,
            commit_message=f"add {local_zip.name}",
        )

        # 2) 最新版本覆盖 model.zip
        upload_file(
            repo_id=args.hf_repo,
            path_or_fileobj=str(local_zip),
            path_in_repo="model.zip",
            token=token,
            commit_message=f"update model.zip → {local_zip.name}",
        )

        print("✔ Upload done: kept history + updated model.zip")

        env.close()


if __name__ == "__main__":
    main()
