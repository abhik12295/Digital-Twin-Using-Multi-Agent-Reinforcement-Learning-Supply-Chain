from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.simulation.environment import SupplyChainRoutingEnv
from src.utils.config import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def make_env(seed: int = 42) -> Monitor:
    env = SupplyChainRoutingEnv(seed=seed)
    env = Monitor(env)
    return env


def train_ppo_model(
    total_timesteps: int = 10000,
    seed: int = 42,
    model_name: str = "ppo_supply_chain",
) -> PPO:
    env = make_env(seed=seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps)

    model_path = MODEL_DIR / model_name
    model.save(str(model_path))
    return model


def load_ppo_model(model_name: str = "ppo_supply_chain") -> PPO:
    model_path = MODEL_DIR / model_name
    return PPO.load(str(model_path))


def evaluate_ppo_model(
    model: PPO,
    episodes: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    env = SupplyChainRoutingEnv(seed=seed)

    episode_rewards: List[float] = []
    episode_costs: List[float] = []
    episode_service: List[float] = []
    episode_disruption: List[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        total_cost = 0.0
        total_on_time = 0
        total_disruption = 0.0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            total_reward += reward
            total_cost += info["cost"]
            total_on_time += info["on_time"]
            total_disruption += info["disruption_score"]
            steps += 1

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost / max(steps, 1))
        episode_service.append(total_on_time / max(steps, 1))
        episode_disruption.append(total_disruption / max(steps, 1))

    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_cost": float(np.mean(episode_costs)),
        "avg_on_time_rate": float(np.mean(episode_service)),
        "avg_disruption_score": float(np.mean(episode_disruption)),
    }