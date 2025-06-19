"""Gym wrapper for akioi-2048 using the official init()."""

from __future__ import annotations
import numpy as np
from gymnasium import Env, spaces
import akioi_2048

STEP_FN = akioi_2048.step
INIT_FN = akioi_2048.init  # 新的初始棋盘生成器
BOARD_SHAPE = (4, 4)
MAX_TILE = 65536
ACTION_MEANINGS = ["Down", "Right", "Up", "Left"]


class Akioi2048Env(Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        self.observation_space = spaces.Box(
            low=-4, high=MAX_TILE, shape=BOARD_SHAPE, dtype=np.int32
        )
        self.action_space = spaces.Discrete(4)
        self.board = INIT_FN()
        self.score = 0

    # Gymnasium API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = INIT_FN()
        self.score = 0
        return np.array(self.board, dtype=np.int32), {}

    def step(self, action: int):
        new_board, delta, msg = STEP_FN(self.board, int(action))
        terminated = msg != 0
        reward = delta
        self.board = new_board
        self.score += delta
        info = {"msg": msg, "score": self.score}
        return np.array(self.board, dtype=np.int32), reward, terminated, False, info

    def render(self):
        return "\n".join(" ".join(f"{v:5d}" for v in r) for r in self.board)
