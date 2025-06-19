"""Gym wrapper for akioi-2048 — reward = final game score."""

from __future__ import annotations
import numpy as np
from gymnasium import Env, spaces
import akioi_2048

STEP_FN = akioi_2048.step  # (board, action) -> new_board, delta, msg
INIT_FN = akioi_2048.init  # returns new 4×4 board

BOARD_SHAPE = (4, 4)
MAX_TILE = 65536  # env metadata
ACTION_MEANINGS = ["Down", "Right", "Up", "Left"]  # 0 → 3


class Akioi2048Env(Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-4, high=MAX_TILE, shape=BOARD_SHAPE, dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self.board: list[list[int]] = INIT_FN()
        self.score: int = 0

    # ───────── Gymnasium API ─────────
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.board = INIT_FN()
        self.score = 0
        return np.asarray(self.board, dtype=np.int32), {}

    def step(self, action: int):
        new_board, delta, msg = STEP_FN(self.board, int(action))
        self.board = new_board
        self.score += delta

        terminated = msg != 0  # 1=win, -1=game over
        reward = self.score if terminated else 0
        info = {"msg": msg, "score": self.score}

        return (
            np.asarray(self.board, dtype=np.int32),
            reward,
            terminated,
            False,  # truncated
            info,
        )

    def render(self):
        return "\n".join(" ".join(f"{v:5d}" for v in row) for row in self.board)
