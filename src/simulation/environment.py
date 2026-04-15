from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from src.utils.config import DATA_PROCESSED_DIR


@dataclass
class StepResult:
    reward: float
    cost: float
    on_time: int
    disruption_score: float


class SupplyChainRoutingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_length: int = 50, seed: int = 42):
        super().__init__()
        self.episode_length = episode_length
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_step = 0
        self.state = None
        self.fred_df = self._load_fred_features()
        self.weather_df = self._load_weather_features()

    def _load_fred_features(self) -> pd.DataFrame:
        path = DATA_PROCESSED_DIR / "fred_features.csv"
        if not path.exists():
            raise FileNotFoundError("fred_features.csv not found. Run main.py first.")
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        return df.dropna().reset_index(drop=True)

    def _load_weather_features(self) -> pd.DataFrame:
        path = DATA_PROCESSED_DIR / "weather_features.csv"
        if not path.exists():
            raise FileNotFoundError("weather_features.csv not found. Run main.py first.")
        return pd.read_csv(path)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.state = self._sample_state_from_live_data()
        return self.state, {}

    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        if max_val == min_val:
            return 0.0
        return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))

    def _sample_state_from_live_data(self) -> np.ndarray:
        fred_row = self.fred_df.sample(1, random_state=int(self.rng.integers(0, 1_000_000))).iloc[0]
        weather_row = self.weather_df.sample(1, random_state=int(self.rng.integers(0, 1_000_000))).iloc[0]

        truck_cost = self._normalize(
            fred_row["PCU484484"],
            self.fred_df["PCU484484"].min(),
            self.fred_df["PCU484484"].max(),
        )
        truckload = self._normalize(
            fred_row["PCU484121484121"],
            self.fred_df["PCU484121484121"].min(),
            self.fred_df["PCU484121484121"].max(),
        )
        local_freight = self._normalize(
            fred_row["PCU484110484110P"],
            self.fred_df["PCU484110484110P"].min(),
            self.fred_df["PCU484110484110P"].max(),
        )

        weather_text = str(weather_row["short_forecast"]).lower()
        disruption_score = 0.2
        if "storm" in weather_text or "snow" in weather_text or "thunder" in weather_text:
            disruption_score = 0.9
        elif "rain" in weather_text or "showers" in weather_text:
            disruption_score = 0.6
        elif "cloud" in weather_text:
            disruption_score = 0.35

        return np.array(
            [truck_cost, truckload, local_freight, disruption_score],
            dtype=np.float32,
        )

    def step(self, action: int):
        assert self.state is not None, "Environment must be reset before stepping."

        truck_cost, truckload, local_freight, disruption = self.state

        #route_efficiency = [1.00, 0.93, 0.88][action]
        if action == 0:  # cheapest
            route_efficiency = 1.05
            disruption_penalty = 1.2

        elif action == 1:  # balanced
            route_efficiency = 0.95
            disruption_penalty = 1.0

        else:  # safest
            route_efficiency = 0.85
            disruption_penalty = 0.7
        effective_disruption = disruption * disruption_penalty

        cost = ((0.5 * truck_cost) + (0.3 * truckload) + (0.2 * local_freight)) / route_efficiency
        cost += float(self.rng.normal(0, 0.05))
        cost = max(cost, 0.0)

        service_risk = (
            (0.65 * disruption * disruption_penalty)
            + (0.20 * truckload)
            + (0.15 * truck_cost)
        )
        # service_risk = 0.65 * disruption + 0.20 * truckload + 0.15 * truck_cost - (0.07 * action)
        on_time = int(service_risk < 0.62)

        reward = (1.7 * on_time) - (1.3 * cost) - (1.1 * effective_disruption)
        result = StepResult(
            reward=float(reward),
            cost=float(cost),
            on_time=on_time,
            disruption_score=float(effective_disruption),
        )

        self.current_step += 1
        terminated = self.current_step >= self.episode_length
        truncated = False
        self.state = self._sample_state_from_live_data() if not terminated else None

        info: Dict[str, float] = {
            "cost": result.cost,
            "on_time": result.on_time,
            "disruption_score": result.disruption_score,
        }
        observation = self.state if self.state is not None else np.zeros(4, dtype=np.float32)
        return observation, result.reward, terminated, truncated, info