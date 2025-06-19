# akioi‑2048‑RL

A cross‑device (CPU & Apple Silicon) reinforcement‑learning pipeline that trains a **DQN** agent to play the [`akioi‑2048`](https://pypi.org/project/akioi-2048/) game.

## Features

* No CUDA required – runs on Intel/AMD CPUs and Apple M‑series GPUs (MPS backend)
* Parallel environment sampling (`SubprocVecEnv`)
* Automatic checkpointing & seamless resume
* One‑command upload/restore from **Hugging Face Hub**

---

## Installation

```bash
uv sync
```

---

## Training – quick start

```bash
python train.py \
  --n-envs 4 \
  --timesteps 1_000_000 \
  --hf-repo jasonxue1/akioi-2048-dqn
```

To **resume** and add more steps:

```bash
python train.py \
  --n-envs 4 \
  --timesteps 2_000_000 \
  --resume jasonxue1/akioi-2048-dqn \
  --hf-repo jasonxue1/akioi-2048-dqn
```

---

## Command‑line parameters (train.py)

| Flag                | Default     | Purpose                                                                                                                                       | When to tune                                                                   |
| ------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `--n-envs`          | `8`         | **Number of parallel game environments**. Each env runs in its own process via `SubprocVecEnv`, boosting sample throughput.                   | Set to 2‑3 × CPU physical cores.<br>M‑chip 4 → use 4, Intel 8‑core → use 8‑16. |
| `--timesteps`       | `2_000_000` | **Steps to add this run**. Passed to `model.learn(total_timesteps=…)`. Stops when done.                                                       | Increase for longer training; typical full run ≥ 5 M steps.                    |
| `--checkpoint-freq` | `250_000`   | **How often to save a checkpoint** (env steps). Internally divided by `n‑envs` to get callback call count.                                    | Lower if you fear power loss; raise to reduce I/O.                             |
| `--resume`          | *None*      | **Resume from an existing model**. Accepts:<br>• Local `.zip` (`checkpoints/…`)<br>• HF repo ID (`user/repo`) – downloads latest `model.zip`. | Use whenever you want to continue an old run.                                  |
| `--hf-repo`         | *None*      | **Push results to Hugging Face Hub**. Provide `username/repo`. Automatically commits new `model.zip`, a 30‑s demo video and a model card.     | Add in every run if you share progress with team / cloud.                      |

### Core hyper‑parameters (in code)

| Variable                 | Location        | Default         | Typical range  | Notes                          |
| ------------------------ | --------------- | --------------- | -------------- | ------------------------------ |
| `learning_rate`          | `DQN(…)`        | `2.5e‑4`        | `1e‑4 – 5e‑4`  | Adam LR for policy & Q‑net     |
| `buffer_size`            | ″               | `1_000_000`     | `5e5 – 2e6`    | Replay buffer capacity         |
| `batch_size`             | ″               | `1024`          | `256 – 1024`   | Gradient batch                 |
| `target_update_interval` | ″               | `10_000`        | `5 k – 20 k`   | How often to sync target Q‑net |
| `net_arch`               | `policy_kwargs` | `[512,512,256]` | any list\[int] | Hidden‑layer sizes             |

Modify these directly in **`train.py`** if you need custom behaviour.

---

## Evaluate

```bash
python eval.py --model jasonxue1/akioi-2048-dqn --episodes 200 --deterministic
```

Outputs per‑episode scores and the mean over `N` episodes.

---

## Directory layout

```
.
├─ akioi_gym.py        # Gymnasium wrapper
├─ train.py            # training entry‑point
├─ eval.py             # evaluation script
├─ pyproject.toml      # deps (CPU + MPS only)
└─ checkpoints/        # auto‑created; rolling Zips + buffers
```

---

