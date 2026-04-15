from __future__ import annotations

import random


class RandomRoutingAgent:
    def __init__(self, action_size: int, seed: int = 42) -> None:
        self.action_size = action_size
        random.seed(seed)

    def act(self) -> int:
        return random.randrange(self.action_size)